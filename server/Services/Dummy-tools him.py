# server/services/dgen_toolshim.py
from __future__ import annotations
import json, re, uuid
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

# ---------------- TOOL HEADER ----------------
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


# ---------------- Message Rendering ----------------
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


# ---------------- Tool Prompt Generator ----------------
def _tools_to_prompt(tools: List[Any]) -> str:
    items = []
    for t in tools or []:
        name = getattr(t, "name", None) or ""
        desc = getattr(t, "description", "").strip()
        schema = getattr(t, "args_schema", None)
        schema_json = json.dumps(schema.schema() if schema else {}, ensure_ascii=False)
        items.append({"name": name, "description": desc, "parameters": schema_json})
    return "TOOLS_SPEC:\n" + json.dumps(items, indent=2, ensure_ascii=False)


_JSON_BLOCK = re.compile(r"({.+})", re.DOTALL)


def _try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        obj = json.loads(match.group(1))
        if isinstance(obj, dict) and "tool_name" in obj:
            return obj
    except Exception:
        return None


# ---------------- Wrapper Class ----------------
class ToolBoundSocgen:
    """Wraps the DGen LLM to support bind_tools() and return AIMessage outputs."""

    def __init__(self, client: Any = None, tools: Optional[List[Any]] = None):
        self.client = client
        self.tools = tools or []

    def bind_tools(self, tools: List[Any]) -> ToolBoundSocgen:
        return ToolBoundSocgen(client=self.client, tools=tools)

    def _call_client(self, prompt: str) -> str:
        """Unified call handler for OpenAI/Anthropic-style clients."""
        try:
            if hasattr(self.client, "invoke"):
                out = self.client.invoke(prompt)
                return out
            if hasattr(self.client, "generate"):
                return self.client.generate(prompt)
            if hasattr(self.client, "__call__"):
                return self.client(prompt)
            raise RuntimeError("LLM client has no valid invoke/generate method.")
        except Exception as e:
            return f"[Error invoking model: {e}]"

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        tool_choice = kwargs.get("tool_choice")
        disable_tools = tool_choice == "none"

        rendered = _render_messages(messages)
        prompt = TOOL_HEADER + "\n" + _tools_to_prompt(self.tools) + "\n" + rendered
        text = self._call_client(prompt)

        if not disable_tools and self.tools:
            parsed = _try_parse_tool_call(text)
            if parsed:
                return AIMessage(content=text, tool_calls=[parsed])

        return AIMessage(content=text)
