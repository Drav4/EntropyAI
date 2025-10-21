# server/services/dgen_toolshim.py
from __future__ import annotations
import json, re, uuid
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

TOOL_HEADER = (
    "You can call tools. When a tool is needed, reply with ONLY a single JSON object and nothing else.\n"
    'Schema: {"tool_name":"<name>","arguments":{...}}\n\n'
    "Examples:\n"
    'User: "Compute dataset facts for data.csv"\n'
    'Assistant:\n{"tool_name":"compute_dataset_facts","arguments":{"file_id_or_name":"data.csv"}}\n\n'
    'User: "Plot histogram of age from data.csv"\n'
    'Assistant:\n{"tool_name":"plot_histogram","arguments":{"file_id_or_name":"data.csv","column":"age","bins":30}}\n\n'
    "If no tool is needed, answer normally in plain text.\n"
)

def _render_messages(messages: List[BaseMessage]) -> str:
    lines: List[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"[SYSTEM] {m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"[USER] {m.content}")
        else:
            lines.append(f"[ASSISTANT] {getattr(m, 'content', '') or ''}")
    return "\n".join(lines)

def _tools_to_prompt(tools: List[Any]) -> str:
    items = []
    for t in tools or []:
        name = getattr(t, "name", None) or t.__name__
        desc = (getattr(t, "description", "") or "").strip()
        schema = getattr(t, "args_schema", None)
        if schema and getattr(schema, "schema", None):
            schema_json = schema.schema()
        else:
            schema_json = {"type": "object", "properties": {}}
        items.append({"name": name, "description": desc, "parameters": schema_json})
    return "TOOLS_SPEC:\n" + json.dumps(items, ensure_ascii=False)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACED = re.compile(r"(\{.*\})", re.DOTALL)

def _try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_BLOCK.search(text) or _BRACED.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if isinstance(obj, dict) and "tool_name" in obj and "arguments" in obj:
        return {"name": str(obj["tool_name"]), "args": obj["arguments"] if isinstance(obj["arguments"], dict) else {}}
    return None

class ToolBoundDGen:
    """Wraps your DGen LLM to support .bind_tools() and return AIMessage/tool_calls."""

    def __init__(self, client: Any, tools: Optional[List[Any]] = None):
        self.client = client     # your DGenAILLM
        self._tools = tools or []

    def bind_tools(self, tools: List[Any]) -> "ToolBoundDGen":
        return ToolBoundDGen(self.client, tools=tools)

    def _call_client(self, prompt: str) -> str:
        # Your DGenAILLM is an LLM → often supports .invoke(prompt) -> str
        if hasattr(self.client, "invoke"):
            return self.client.invoke(prompt)
        if hasattr(self.client, "__call__"):
            return self.client(prompt)
        if hasattr(self.client, "generate"):
            out = self.client.generate([prompt])
            try:
                return out.generations[0][0].text
            except Exception:
                return str(out)
        raise RuntimeError("DGen client has no .invoke/.generate/__call__")

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        sys_hdr = TOOL_HEADER + "\n" + _tools_to_prompt(self._tools) if self._tools else ""
        rendered = _render_messages(messages)
        prompt = (sys_hdr + "\n" + rendered) if sys_hdr else rendered

        text = self._call_client(prompt)

        if self._tools:
            parsed = _try_parse_tool_call(text)
            if parsed:
                return AIMessage(content="", tool_calls=[{
                    "name": parsed["name"], "args": parsed["args"], "id": str(uuid.uuid4())
                }])

        return AIMessage(content=text)
