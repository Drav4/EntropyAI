# server/core/toolsim.py
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from pydantic import BaseModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ---------------------------------------------------------------------
# Tool call protocol prompt (model-facing instruction)
# ---------------------------------------------------------------------
TOOL_HEADER = (
    "You can call tools. When a tool is needed, reply with ONLY a single JSON object and nothing else.\n"
    'Schema: {"tool_name":"<name>","arguments":{...}}\n\n'
    "If a dataset was previously referenced, assume the same dataset unless the user specifies another.\n"
    "Examples:\n"
    'User: "Compute dataset facts for data.csv"\n'
    'Assistant:\n{"tool_name":"compute_dataset_facts","arguments":{"file_id_or_name":"data.csv"}}\n\n'
    'User: "Plot histogram for column age"\n'
    'Assistant:\n{"tool_name":"plot_histogram","arguments":{"file_id_or_name":"data.csv","column":"age","bins":30}}\n\n'
    "If no tool is needed, answer normally in plain text.\n"
)

# ---------------------------------------------------------------------
# Render chat messages into a plain text transcript for the model
# ---------------------------------------------------------------------
def _render_messages(messages: List[BaseMessage]) -> str:
    lines: List[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"[SYSTEM] {m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"[USER] {m.content}")
        elif isinstance(m, ToolMessage):
            tool_name = getattr(m, "name", getattr(m, "tool", None))
            lines.append(f"[ToolCall:{tool_name}] {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"[ASSISTANT] {m.content}")
        else:
            lines.append(f"[ASSISTANT] {getattr(m, 'content', '') or ''}")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# Serialize tool specs (name, description, JSON schema)
# ---------------------------------------------------------------------
def _schema_for_args(args_schema: Optional[Type[BaseModel]]) -> Dict[str, Any]:
    if not args_schema:
        return {}
    # Pydantic v1 & v2 support
    try:
        return args_schema.schema()  # type: ignore[attr-defined]
    except Exception:
        try:
            return args_schema.model_json_schema()  # pydantic v2
        except Exception:
            return {}

def _tools_to_prompt(tools: List[Any]) -> str:
    items = []
    for t in tools or []:
        name = getattr(t, "name", "") or ""
        desc = (getattr(t, "description", "") or "").strip()
        args_schema = getattr(t, "args_schema", None)
        schema_json = _schema_for_args(args_schema)
        items.append({"name": name, "description": desc, "parameters": schema_json})
    return "TOOLS_SPEC:\n" + json.dumps(items, ensure_ascii=False)

# ---------------------------------------------------------------------
# Extract a JSON object from model text and normalize keys
# ---------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"\{[\s\S]*?\}", re.DOTALL)

def _try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Accepts either:
      {"tool_name":"x","arguments":{...}}  (per TOOL_HEADER)
    or:
      {"name":"x","args":{...}}            (already normalized)
    Returns normalized: {"name":"x","args":{...}}
    """
    if not isinstance(text, str) or not text:
        return None
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("tool_name")
        args = obj.get("args") or obj.get("arguments") or {}
        if name and isinstance(args, dict):
            return {"name": name, "args": args}
    return None

# ---------------------------------------------------------------------
# Tool adapter used by your registry
# ---------------------------------------------------------------------
class ToolWrap:
    """
    Wraps a Python callable with a name/description and an optional Pydantic args schema.
    Registry will create these and pass them to ToolBound.
    """
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Optional[Type[BaseModel]],
        fn: Callable[..., Any],
        how_computed: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self._fn = fn
        self.how_computed = how_computed or ""

    def invoke(self, args: Dict[str, Any]) -> Any:
        # Assume args already validated by model prompt; Pydantic validation is optional at registry level
        return self._fn(**(args or {}))

# ---------------------------------------------------------------------
# ToolBound: binds tools to an LLM client and parses tool calls
# ---------------------------------------------------------------------
class ToolBound:
    """
    Binds a raw LLM client to a set of tools.
    - Renders TOOL_HEADER + TOOLS_SPEC + conversation transcript.
    - Calls underlying client (.invoke/.generate/callable).
    - Parses a single JSON tool call if present; returns AIMessage with .tool_calls.
    - Can execute a requested tool via invoke_tool().
    """

    def __init__(self, client: Any = None, tools: Optional[List[Any]] = None):
        self.client = client
        self.tools = tools or []

    def bind_tools(self, tools: List[Any]) -> "ToolBound":
        return ToolBound(client=self.client, tools=tools)

    # ---- LLM call adapter (robust normalization to string) ----
    def _call_client(self, prompt: str) -> str:
        """
        Call the underlying LLM client and normalize the output to a string.
        Handles LangChain AIMessage, dicts, and providers that expose .text as a method.
        """
        try:
            if hasattr(self.client, "invoke"):
                out = self.client.invoke(prompt)
            elif hasattr(self.client, "generate"):
                out = self.client.generate(prompt)
            elif callable(self.client):
                out = self.client(prompt)
            else:
                return ""
        except Exception as e:
            return f"[Model error: {e}]"

        # 1) LangChain message objects (AIMessage, etc.)
        content = getattr(out, "content", None)
        if isinstance(content, str):
            return content

        # 2) Some wrappers expose .text (string or callable)
        txt = getattr(out, "text", None)
        if txt is not None:
            if callable(txt):
                try:
                    txt_val = txt()
                    if isinstance(txt_val, str):
                        return txt_val
                except Exception:
                    pass
            elif isinstance(txt, str):
                return txt

        # 3) Dict-style responses
        if isinstance(out, dict):
            for k in ("content", "text", "message", "output"):
                v = out.get(k)
                if isinstance(v, str):
                    return v
            try:
                return json.dumps(out, ensure_ascii=False)
            except Exception:
                return str(out)

        # 4) Fallback to string repr
        return str(out)

    # ---- Main entry: produce next assistant message (maybe with tool_calls) ----
# server/core/toolsim.py  (inside class ToolBound)

    def invoke(self, messages, **kwargs):
        tool_choice = kwargs.get("tool_choice", None)
        disable_tools = (tool_choice == "none")

        rendered = _render_messages(messages)
        prompt = TOOL_HEADER + "\n" + _tools_to_prompt(self.tools) + "\n" + rendered

        text = self._call_client(prompt)
        if not isinstance(text, str):
            text = str(text)

        if not disable_tools and self.tools:
            parsed = _try_parse_tool_call(text)
            if parsed:
                tc = {"name": parsed["name"], "args": parsed["args"], "id": str(uuid.uuid4())}
                # ✅ CRITICAL: do not leak JSON in content
                return AIMessage(content="", tool_calls=[tc])

        return AIMessage(content=text or "")


    # ---- Execute a tool call and return a ToolMessage ----
    def invoke_tool(self, call: Dict[str, Any], state: Dict[str, Any]) -> ToolMessage:
        """
        call = {"name": "...", "args": {...}}
        """
        name = call.get("name")
        args = call.get("args", {}) or {}

        # Find tool implementation
        tool_impl: Optional[ToolWrap] = None
        for t in (self.tools or []):
            if getattr(t, "name", None) == name:
                tool_impl = t
                break

        if tool_impl is None:
            return ToolMessage(
                name=name or "tool",
                content=json.dumps({"ok": False, "error": f"Unknown tool '{name}'", "arguments": args})
            )

        try:
            result = tool_impl.invoke(args)
            payload = result if isinstance(result, dict) else {"result": result}
            # Add arguments & provenance for downstream writer/evidence
            payload.setdefault("arguments", args)
            if getattr(tool_impl, "how_computed", None):
                payload.setdefault("how_computed", tool_impl.how_computed)
            return ToolMessage(name=name, tool_call_id=str(uuid4()),content=json.dumps(payload))
        except Exception as e:
            return ToolMessage(
                name=name or "tool",
                tool_call_id=str(uuid4()),
                content=json.dumps({
                    "ok": False,
                    "error": f"Tool '{name}' failed: {e}",
                    "arguments": args
                }),
            )
