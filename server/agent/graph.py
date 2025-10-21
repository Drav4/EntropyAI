# server/agent/graph.py

from __future__ import annotations

from typing import TypedDict, Literal, List, Optional, Dict, Any

import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from ..services.openai_client import make_llm  # your LLM factory
from .tools_langgraph import TOOLS             # list of @tool(...) tools
from ..config import UPLOAD_DIR


# -----------------------------
# Agent state
# -----------------------------
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]          # full chat history
    attachments: List[str]               # file IDs / names (uploads/)
    must_finalize: bool                  # after tools, force a "no-tools" answer
    steps: int                           # safety counter


MAX_STEPS = 12


# -----------------------------
# Helpers
# -----------------------------
def _exists_in_uploads(fid: Optional[str]) -> bool:
    if not fid:
        return False
    path = fid if os.path.isabs(fid) else os.path.join(UPLOAD_DIR, fid)
    return os.path.exists(path)


# -----------------------------
# Nodes
# -----------------------------
def llm_node(state: AgentState) -> AgentState:
    """
    Core LLM node.
    - Grounds the current attachment so the model uses it.
    - If tools were just run, performs a finalization pass with tools disabled.
    - If the model returns tool_calls, returns an AIMessage with EMPTY content
      so the UI never sees a JSON blob.
    """
    llm = make_llm().bind_tools(TOOLS)

    steps = state.get("steps", 0) + 1
    if steps > MAX_STEPS:
        return {"messages": [AIMessage(content="Stopping: reasoning limit reached.")], "steps": steps}

    msgs: List[BaseMessage] = list(state.get("messages", []))

    # Ground the current attachment (if any) so the model doesn't invent filenames
    if state.get("attachments"):
        current = state["attachments"][0]
        msgs.append(
            HumanMessage(
                content=f"[Attachments]\n#1: {current} (id:{current})\n"
                        f"Use this file for any dataset operations."
            )
        )

    # Finalization pass: answer using tool output; do NOT call tools again
    if state.get("must_finalize"):
        res = llm.invoke(msgs, tool_choice="none")
        return {"messages": [res], "must_finalize": False, "steps": steps}

    # Normal reasoning turn (tools allowed)
    res = llm.invoke(msgs)

    # If the model produced tool calls, strip any textual JSON from content
    # so the UI doesn't render it; keep only tool_calls.
    tool_calls = getattr(res, "tool_calls", None)
    if tool_calls:
        res = AIMessage(content="", tool_calls=tool_calls)

    return {"messages": [res], "steps": steps}


def route_after_llm(state: AgentState) -> Literal["tools", "final"]:
    """
    Router: if the last assistant message contains tool_calls, go run tools.
    Otherwise, we're done.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    return "tools" if tool_calls else "final"


def tools_node():
    """
    Tool runner with argument grounding:
    - If a tool call is missing/has a bad file_id_or_name, replace it with the
      currently attached file before executing the tool.
    """
    base = ToolNode(TOOLS)

    def run(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", []) or []
        for c in calls:
            args: Dict[str, Any] = c.get("args", {})
            fid = args.get("file_id_or_name")

            # Ground to attachment if the model omitted or guessed a stale name
            if (not _exists_in_uploads(fid)) and state.get("attachments"):
                args["file_id_or_name"] = state["attachments"][0]

        return base.invoke(state)

    return run


def after_tools(state: AgentState) -> AgentState:
    """
    After tools: mark that we must finalize in the next LLM turn.
    """
    return {"must_finalize": True}


# -----------------------------
# Graph builder
# -----------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node())
    g.add_node("after_tools", after_tools)

    g.set_entry_point("llm")

    # LLM → tools when tool_calls; otherwise finish
    g.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "final": END})

    # Tools → after_tools → LLM (finalization, tool_choice="none")
    g.add_edge("tools", "after_tools")
    g.add_edge("after_tools", "llm")

    return g.compile()
