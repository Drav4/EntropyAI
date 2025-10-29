# server/agent/nodes/tool_node.py
from __future__ import annotations
import json
from typing import Any, Dict
from langchain_core.messages import ToolMessage
from ...core.toolsim import ToolBound
from ...tools.registry import TOOLS
from ..state import GraphState

_runner = ToolBound(tools=TOOLS)

def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    s = s.strip()
    # Normal JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            # double-encoded JSON
            try:
                inner = json.loads(obj)
                return inner if isinstance(inner, dict) else {}
            except Exception:
                pass
    except Exception:
        pass
    # Extract first {...} block
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            inner = json.loads(s[start:end + 1])
            return inner if isinstance(inner, dict) else {}
        except Exception:
            return {}
    return {}

def _merge_facts(facts: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(facts.get(k), dict):
            facts[k].update(v)
        else:
            facts[k] = v
    return facts

def tool_node(state: GraphState) -> GraphState:
    if not state.tool_calls:
        return state

    for call in state.tool_calls:
        tmsg: ToolMessage = _runner.invoke_tool(call, state=state.dict())
        state.messages.append(tmsg)

        raw = tmsg.content or ""
        print("\n[tool_node] RAW:", raw[:400])  # DEBUG

        payload = _safe_json_loads(raw)
        print("[tool_node] PARSED KEYS:", list(payload.keys()))  # DEBUG

        if payload:
            state.facts = _merge_facts(state.facts or {}, payload)
            state.evidence.setdefault(call["name"], []).append(payload)

    state.tool_calls = []
    state.steps += 1
    return state
