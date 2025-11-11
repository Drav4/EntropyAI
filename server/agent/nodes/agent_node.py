from __future__ import annotations
import re
from langchain_core.messages import AIMessage, SystemMessage
from ...llm import make_llm
from ...core.toolsim import ToolBound
from ...tools.registry import TOOLS
from ..state import GraphState

AGENT_SYSTEM = r"""
You alternate between two modes:

[TOOL_MODE]
- Output EXACTLY one JSON object to call a tool:
  {"tool_name":"<compute_dataset_facts|compute_correlation|plot_histogram|profile_for_model_selection>",
   "arguments":{...}}
- Use the active dataset id mentioned in system messages if present.
- If a target/label is required but not provided, DO NOT guess. Ask one concise question.

[READY_MODE]
- If enough facts are present, output exactly: READY_FOR_WRITER

CONCEPT GATE:
- If the latest user request is a conceptual DS/ML question (definitions, comparisons,
  theory, pros/cons, “what is…”, “explain…”, “compare…”), DO NOT call any tools.
  Instead, say READY_FOR_WRITER so the writer can answer.
- If the request is clearly outside DS/ML (handled upstream by guard), also say READY_FOR_WRITER.

Do NOT produce the final user-facing answer. The writer will do that.
"""

_agent_llm = make_llm(system_prompt=AGENT_SYSTEM, temperature=0.1)
_agent = ToolBound(client=_agent_llm, tools=TOOLS)

# ---- NEW: robust target/file extraction ----
_TARGET_PAT = re.compile(
    r"\btarget(?:\s*column)?\s*(?:=|:|is)\s*[\"'`]?([\w\s\-]+?)[\"'`]?(?:[^\w]|$)",
    re.IGNORECASE,
)

def _extract_target(messages) -> str | None:
    # Search newest → oldest; accept "target is", "target:", "target ="
    for m in reversed(messages):
        txt = (getattr(m, "content", "") or "").strip()
        mt = _TARGET_PAT.search(txt)
        if mt:
            return mt.group(1).strip()
    return None

def _extract_active_file(messages) -> str | None:
    # We inject a SystemMessage: Active dataset id: '<id>'
    for m in messages:
        if isinstance(m, SystemMessage) and "Active dataset id:" in (m.content or ""):
            tail = m.content.split("Active dataset id:", 1)[1].strip()
            return tail.strip("'\" ")
    return None

def agent_node(state: GraphState) -> GraphState:
    """
    Preflight guarantees:
      - If file present & target present  -> enqueue profile_for_model_selection
      - If file present & target missing -> enqueue compute_dataset_facts
      - Else                              -> let LLM decide
    """
    file_id = _extract_active_file(state.messages)
    target = _extract_target(state.messages)

    if file_id and not state.tool_calls and not state.facts:
        if target:
            state.tool_calls = [{
                "name": "profile_for_model_selection",
                "args": {"file_id_or_name": file_id, "target": target},
                "id": "preflight-1"
            }]
            state.steps += 1
            return state
        else:
            state.tool_calls = [{
                "name": "compute_dataset_facts",
                "args": {"file_id_or_name": file_id},
                "id": "preflight-2"
            }]
            state.steps += 1
            return state

    # Normal agent pass
    result: AIMessage = _agent.invoke(state.messages)
    state.messages.append(result)
    state.tool_calls = getattr(result, "tool_calls", []) or []
    state.steps += 1
    return state
