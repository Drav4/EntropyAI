# server/agent/graph.py
import os, json
from typing import TypedDict, List, Literal, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)

from tools_langgraph import make_llm, TOOLS  # ✅ keep your factory
from config import UPLOAD_DIR


# ---------- Agent State ----------
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], "Messages exchanged with LLM"]
    grounded: bool
    must_finalize: bool
    last_tool_name: Optional[str]
    last_dataset: Optional[str]            # ✅ persist dataset between turns
    step: int


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
    step = int(state.get("step", 0)) + 1
    if step > _MAX_STEPS:
        return {"messages": [AIMessage(content="I reached my reasoning limit while looping over tools.")]}

    llm = make_llm().bind_tools(TOOLS)  # ✅ your factory
    msgs = state.get("messages", [])
    out = llm.invoke(msgs, tool_choice=None)  # allow tools by default

    return {
        "messages": msgs + [out],
        "step": step,
        "must_finalize": False,
    }


# ---------- Tool Node ----------
def tool_node(state: AgentState) -> AgentState:
    """
    Runs the tool indicated by the last assistant tool call and appends the ToolMessage.
    Also injects/remember file_id_or_name so the first histogram works.
    """
    from services.socgen_toolshim import ToolBoundSocgen  # local import to avoid cycles
    base = ToolBoundSocgen()

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {"messages": state["messages"]}  # nothing to do

    call = dict(tool_calls[0])  # shallow copy

    # ---- dataset grounding / persistence ----
    args = dict(call.get("args") or call.get("arguments") or {})
    fid = args.get("file_id_or_name")

    # 1) fallback to last remembered dataset
    if not fid:
        fid = state.get("last_dataset")

    # 2) fallback to first uploaded attachment (if you populate state['attachments'])
    if not fid and state.get("attachments"):
        fid = state["attachments"][0]

    # 3) persist and inject back
    if fid:
        args["file_id_or_name"] = fid
        state["last_dataset"] = fid

    # 4) guard invalid dataset early
    if not _exists_in_uploads(fid):
        tm = ToolMessage(
            name=call.get("name") or "tool",
            content=json.dumps({
                "ok": False,
                "error": "Invalid or missing dataset reference.",
                "arguments": args
            })
        )
        return {"messages": state["messages"] + [tm], "must_finalize": True}

    # normalize call obj (your shim accepts this shape)
    call["args"] = args

    # ---- invoke the tool via your shim ----
    res_msg = base.invoke_tool(call, state)  # must return a ToolMessage
    return {"messages": state["messages"] + [res_msg], "must_finalize": False}


# ---------- After Tools ----------
def after_tools(state: AgentState) -> AgentState:
    """
    Post-tool updates: mark grounded where appropriate and persist last_dataset.
    """
    last_msg = state["messages"][-1]
    grounded = bool(state.get("grounded", False))
    last_tool_name = getattr(last_msg, "name", None)

    # if you use compute_dataset_facts as your "grounding" tool, keep this:
    if last_tool_name == "compute_dataset_facts":
        grounded = True

    new_state: AgentState = dict(state)
    new_state.update({
        "grounded": grounded,
        "must_finalize": False,
        "last_tool_name": last_tool_name,
    })

    # Try to persist dataset from tool output payload too
    try:
        if isinstance(last_msg, ToolMessage):
            payload = json.loads(last_msg.content)
            fid = (payload.get("arguments") or {}).get("file_id_or_name")
            if fid:
                new_state["last_dataset"] = fid
    except Exception:
        pass

    return new_state


# ---------- Routing ----------
def route_after_llm(state: AgentState) -> Literal["tools", "final"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    return "tools" if tool_calls else "final"


# ---------- Graph Builder ----------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node)
    g.add_node("after_tools", after_tools)

    g.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "final": END})
    g.add_edge("tools", "after_tools")
    g.add_edge("after_tools", "llm")

    g.set_entry_point("llm")
    return g.compile()
