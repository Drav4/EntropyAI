# server/agent/graph.py
import os, json
from typing import TypedDict, List, Literal, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from services.dgen_toolshim import ToolBoundSocgen
from tools_langgraph import TOOLS
from config import UPLOAD_DIR


# ---------- Agent State ----------
class AgentState(TypedDict, total=False):
    messages: Annotated[List, "Messages exchanged with LLM"]
    grounded: bool
    must_finalize: bool
    last_tool_name: Optional[str]
    last_dataset: Optional[str]  # 👈 Persist dataset context


_MAX_STEPS = 8


# ---------- Utils ----------
def _exists_in_uploads(fid: Optional[str]) -> bool:
    if not fid:
        return False
    path = fid if os.path.isabs(fid) else os.path.join(UPLOAD_DIR, fid)
    return os.path.exists(path)


# ---------- LLM Node ----------
def llm_node(state: AgentState) -> AgentState:
    """Core LLM node."""
    llm = ToolBoundSocgen().bind_tools(TOOLS)
    step = state.get("step", 0) + 1
    if step > _MAX_STEPS:
        return {
            "messages": [AIMessage(content="Reached reasoning limit while looping over tools.")],
        }

    # Compose conversation
    msgs = state.get("messages", [])
    out = llm.invoke(msgs, tool_choice="none", disable_tools=False)
    return {
        "messages": msgs + [out],
        "step": step,
        "must_finalize": False,
    }


# ---------- Tool Node ----------
def tool_node(state: AgentState) -> AgentState:
    """Runs tool and appends ToolMessage."""
    base = ToolBoundSocgen()
    last = state["messages"][-1]
    call = getattr(last, "tool_calls", None) or [{}]
    call = call[0]

    fid = call.get("arguments", {}).get("file_id_or_name")

    # --- Fix: fallback to last dataset if not provided
    if not fid:
        fid = state.get("last_dataset")

    # --- Fallback to first uploaded attachment if still missing
    if not fid and state.get("attachments"):
        fid = state["attachments"][0]

    # --- Persist dataset reference
    if fid:
        call["arguments"]["file_id_or_name"] = fid
        state["last_dataset"] = fid

    # --- Skip invalid attachment
    if not _exists_in_uploads(fid):
        return {
            "messages": [
                ToolMessage(content="Invalid or missing dataset reference.", name=call.get("name", "tool"))
            ],
            "must_finalize": True,
        }

    # Invoke tool safely
    res = base.invoke_tool(call, state)
    return {"messages": state["messages"] + [res], "must_finalize": False}


# ---------- After Tools Node ----------
def after_tools(state: AgentState) -> AgentState:
    """Handle post-tool cleanup and memory persistence."""
    last = state["messages"][-1]
    grounded = state.get("grounded", False)
    last_tool_name = getattr(last, "name", None)

    if last_tool_name == "compute_dataset_facts":
        grounded = True

    new_state = dict(state)
    new_state.update(
        {
            "grounded": grounded,
            "step": state.get("step", 0) + 1,
            "must_finalize": False,
            "last_tool_name": last_tool_name,
        }
    )

    # --- Persist last dataset used
    try:
        if isinstance(last, ToolMessage):
            content = json.loads(last.content)
            fid = content.get("arguments", {}).get("file_id_or_name")
            if fid:
                new_state["last_dataset"] = fid
    except Exception:
        pass

    return new_state


# ---------- Routing ----------
def route_after_llm(state: AgentState) -> Literal["tools", "final"]:
    """Route based on whether tool calls exist."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "final"


# ---------- Graph Builder ----------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node)
    g.add_node("after_tools", after_tools)

    # Flow control
    g.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "final": END})
    g.add_edge("tools", "after_tools")
    g.add_edge("after_tools", "llm")

    g.set_entry_point("llm")
    return g.compile()
