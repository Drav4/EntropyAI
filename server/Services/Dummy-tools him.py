from __future__ import annotations
import json, re, uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
)

# ---------------- Prompt header ----------------
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

# ---------------- Render messages for model ----------------
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
        else:
            lines.append(f"[ASSISTANT] {getattr(m, 'content', '') or ''}")
    return "\n".join(lines)

# ---------------- Tools spec serializer ----------------
def _tools_to_prompt(tools: List[Any]) -> str:
    items = []
    for t in tools or []:
        name = getattr(t, "name", "") or ""
        desc = (getattr(t, "description", "") or "").strip()
        schema = getattr(t, "args_schema", None)
        schema_json = (schema.schema() if schema else {})
        items.append({"name": name, "description": desc, "parameters": schema_json})
    return "TOOLS_SPEC:\n" + json.dumps(items, ensure_ascii=False)

# Grab first JSON object even if surrounded by extra text/fences
_JSON_BLOCK = re.compile(r"\{[\s\S]*?\}", re.DOTALL)

def _try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Accepts either:
      {"tool_name":"x","arguments":{...}}  (model per TOOL_HEADER)
    or:
      {"name":"x","args":{...}}            (already normalized)
    Returns normalized: {"name":"x","args":{...}}
    """
    if not text:
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


class ToolBoundSocgen:
    """
    Adapter that:
      * builds the full prompt (system header + tools spec + rendered messages)
      * calls the underlying client (wired in your make_llm())
      * returns AIMessage, optionally with .tool_calls normalized to {"name","args","id"}
    """

    def __init__(self, client: Any = None, tools: Optional[List[Any]] = None):
        self.client = client
        self.tools = tools or []

    def bind_tools(self, tools: List[Any]) -> "ToolBoundSocgen":
        return ToolBoundSocgen(client=self.client, tools=tools)

    # Low-level model call wrapper (works with invoke/generate/callable clients)
    def _call_client(self, prompt: str) -> str:
        try:
            if hasattr(self.client, "invoke"):
                out = self.client.invoke(prompt)
                return out if isinstance(out, str) else getattr(out, "text", str(out))
            if hasattr(self.client, "generate"):
                out = self.client.generate(prompt)
                return getattr(out, "text", str(out))
            if callable(self.client):
                return self.client(prompt)
        except Exception as e:
            return f"[Model error: {e}]"
        return ""

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        tool_choice = kwargs.get("tool_choice", None)  # None -> tools allowed
        disable_tools = tool_choice == "none"

        rendered = _render_messages(messages)
        prompt = TOOL_HEADER + "\n" + _tools_to_prompt(self.tools) + "\n" + rendered
        text = self._call_client(prompt)

        if not disable_tools and self.tools:
            parsed = _try_parse_tool_call(text)
            if parsed:
                # ✅ LangChain expects "name" / "args" / "id" (NOT tool_name/arguments)
                tc = {"name": parsed["name"], "args": parsed["args"], "id": str(uuid.uuid4())}
                return AIMessage(content=text, tool_calls=[tc])

        return AIMessage(content=text)

    def invoke_tool(self, call: Dict[str, Any], state: Dict[str, Any]) -> ToolMessage:
        """
        Graph calls this to run a tool. Expects call = {"name": "...", "args": {...}}
        """
        name = call.get("name")
        args = call.get("args", {}) or {}

        # find tool implementation
        tool_impl = None
        for t in (self.tools or []):
            if getattr(t, "name", None) == name:
                tool_impl = t
                break
        if tool_impl is None:
            return ToolMessage(name=name or "tool", content=json.dumps({
                "ok": False, "error": f"Unknown tool '{name}'", "arguments": args
            }))

        try:
            result = tool_impl.invoke(args) if hasattr(tool_impl, "invoke") else tool_impl(**args)
            payload = result if isinstance(result, dict) else {"result": result}
            # include arguments so after_tools can persist last_dataset
            if "arguments" not in payload:
                payload["arguments"] = args
            return ToolMessage(name=name, content=json.dumps(payload))
        except Exception as e:
            return ToolMessage(name=name, content=json.dumps({
                "ok": False, "error": f"Tool '{name}' failed: {e}", "arguments": args
            }))
