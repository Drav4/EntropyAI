import os
import json
from typing import TypedDict, List, Literal, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages  # if you used this earlier, keep it; else safe to leave
from langgraph.prebuilt import ToolNode

from services.socgen_client import make_llm
from tools_langgraph import TOOLS
from config import UPLOAD_DIR

# ---------- Agent state ----------
class AgentState(TypedDict, total=False):
    # your original used Annotated[ add_messages ] – keep it if you had it
    messages: Annotated[List[BaseMessage], add_messages]
    attachments: List[str]                     # candidate file ids/names from [Attachments]
    grounded: bool                             # True once we've computed dataset facts at least once
    steps: int                                 # safety counter to avoid infinite loops
    must_finalize: bool
    last_tool_name: Optional[str]
    last_dataset: Optional[str]                # ✅ NEW: persist last used dataset

_MAX_STEPS = 8

def _exists_in_uploads(fid: Optional[str]) -> bool:
    if not fid:
        return False
    path = fid if os.path.isabs(fid) else os.path.join(UPLOAD_DIR, fid)
    return os.path.exists(path)

# ---------- LLM node ----------
def llm_node(state: AgentState) -> AgentState:
    """
    Core LLM node.
    - On the first pass (attachments present & not grounded), force a call to 'compute_dataset_facts'.
    - Otherwise: normal tool-calling invocation.
    """
    llm = make_llm().bind_tools(TOOLS)
    msgs = state["messages"]
    state["steps"] = state.get("steps", 0) + 1

    if state["steps"] > 24:
        return {
            "messages": [
                AIMessage(content="I reached my reasoning limit while looping over tools")
            ],
        }

    if state.get("must_finalize"):
        guidance = HumanMessage(content=(
            "Use the [Tool:] result above to answer the user. Do not call any tools again unless the user asks "
            "for a new operation."
        ))
        res = llm.invoke([*msgs, guidance], tool_choice="none")
        return {"messages": [guidance, res], "must_finalize": False}

    # If we have attachments but haven't grounded yet, nudge with a hint and force the tool choice to compute_dataset_facts.
    if state.get("attachments") and not state.get("grounded"):
        hint = HumanMessage(content=f"Available dataset files: {state['attachments']}")
        res = llm.invoke(
            [*msgs, hint],
            tool_choice={"type": "function", "function": {"name": "compute_dataset_facts"}},
        )
        return {"messages": [hint, res]}
    else:
        res = llm.invoke(msgs)
        # if your previous code wrapped tool_calls into an empty-content AIMessage, keep that:
        tool_calls = getattr(res, "tool_calls", None)
        if tool_calls and getattr(res, "content", None):
            res = AIMessage(content="", tool_calls=tool_calls)
        return {"messages": [res]}

# ---------- Router after LLM ----------
def route_after_llm(state: AgentState) -> Literal["tools", "final"]:
    """
    Decide where to go after the LLM:
    - If the last assistant message contains tool_calls -> run tools.
    - Otherwise -> we're done (final).
    """
    last = state["messages"][-1]

    if state.get("steps", 0) > _MAX_STEPS:
        return "final"

    tool_calls = getattr(last, "tool_calls", None)
    return "tools" if tool_calls else "final"

# ---------- Tools node ----------
def tools_node() -> ToolNode:
    """Runs the tools and appends ToolMessages to the state's messages list."""
    base = ToolNode(TOOLS)

    def run(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", []) or []

        if not calls:
            return {"messages": state["messages"]}

        normalized_calls = []
        for c in calls:
            # Accept either {"tool_name","arguments"} or {"name","args"}
            name = c.get("name") or c.get("tool_name")
            args = dict(c.get("args") or c.get("arguments") or {})
            if not name:
                # skip malformed entries
                continue

            # ---- dataset fallback & persistence (fix first histogram) ----
            fid = args.get("file_id_or_name") or state.get("last_dataset")
            if not fid and state.get("attachments"):
                fid = state["attachments"][0]
            if fid:
                args["file_id_or_name"] = fid
                state["last_dataset"] = fid

            # (optional) validate path under uploads
            # if fid and not _exists_in_uploads(fid):
            #     tm = ToolMessage(
            #         name=name,
            #         content=json.dumps({"ok": False, "error": "Invalid or missing dataset reference.", "arguments": args})
            #     )
            #     return {"messages": state["messages"] + [tm], "must_finalize": True}

            normalized_calls.append({
                "name": name,
                "args": args,
                "id": c.get("id") or str(uuid.uuid4()),
            })

        # Replace the last AI message with a normalized one so ToolNode can execute it
        last_ai = AIMessage(content=getattr(last, "content", "") or "", tool_calls=normalized_calls)
        new_msgs = [*state["messages"][:-1], last_ai]
        out = base.invoke({"messages": new_msgs})  # ToolNode expects the same state shape
        return out

    return run

# ---------- After tools ----------
def after_tools(state: AgentState) -> AgentState:
    grounded = state.get("grounded", False)
    last = state["messages"][-1]
    last_tool_name = getattr(last, "name", None) if hasattr(last, "name") else None
    if last_tool_name == "compute_dataset_facts":
        grounded = True

    # persist last_dataset from tool payload if present
    try:
        if isinstance(last, ToolMessage):
            payload = json.loads(last.content)
            fid = (payload.get("arguments") or {}).get("file_id_or_name")
            if fid:
                state["last_dataset"] = fid
    except Exception:
        pass

    new_state = dict(state)  # copy
    new_state.update({
        "grounded": grounded,
        "steps": state.get("steps", 0),
        "must_finalize": False,
        "last_tool_name": last_tool_name,
    })
    return new_state

# ---------- Graph builder ----------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node())
    g.add_node("after_tools", after_tools)

    # After LLM: either go run tools (if tool_calls) or finish.
    g.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "final": END})

    # After tools: mark grounded/step++ then go back to LLM (loop).
    g.add_edge("tools", "after_tools")
    g.add_edge("after_tools", "llm")

    g.set_entry_point("llm")
    return g.compile()
