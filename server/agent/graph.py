# server/agent/graph.py

from typing import TypedDict, List, Literal, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages   # <- reducer for appending messages
from langgraph.prebuilt import ToolNode

from ..services.openai_client import make_llm
from .tools_langgraph import TOOLS


# ---------- Agent state ----------
# Use Annotated[..., add_messages] so tool/assistant messages are APPENDED correctly.
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    attachments: List[str]   # candidate file ids/names from [Attachments]
    grounded: bool           # True once we've computed dataset facts at least once
    steps: int               # safety counter to avoid infinite loops


MAX_STEPS = 8


# ---------- Nodes ----------
def llm_node(state: AgentState) -> AgentState:
    """
    Core LLM node.
    - On the first pass (attachments present & not grounded), force a call to `compute_dataset_facts`.
    - Otherwise: normal tool-calling invocation.
    """
    llm = make_llm().bind_tools(TOOLS)
    msgs = state["messages"]

    # If we have attachments but haven't grounded yet, nudge with a hint
    # and force the tool choice to compute_dataset_facts.
    if state.get("attachments") and not state.get("grounded"):
        hint = HumanMessage(content=f"Available dataset files: {state['attachments']}")
        res = llm.invoke(
            [*msgs, hint],
            tool_choice={"type": "function", "function": {"name": "compute_dataset_facts"}},
        )
        # With add_messages reducer, returning a list appends both hint and res
        return {"messages": [hint, res]}
    else:
        res = llm.invoke(msgs)
        return {"messages": [res]}


def route_after_llm(state: AgentState) -> Literal["tools", "final"]:
    """
    Decide where to go after the LLM:
    - If the last assistant message contains tool_calls -> run tools.
    - Otherwise -> we're done (final).
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "final"


def tools_node() -> ToolNode:
    """Runs the tools and appends ToolMessages to the state's messages list."""
    return ToolNode(TOOLS)


def after_tools(state: AgentState) -> AgentState:
    """
    Post-process tool outputs:
    - If we just computed dataset facts, mark grounded=True.
    - Increment step counter.
    IMPORTANT: Do NOT return 'messages' here, or you'll overwrite the history.
    """
    grounded = state.get("grounded", False)
    last = state["messages"][-1]
    if hasattr(last, "name") and last.name == "compute_dataset_facts":
        grounded = True

    return {
        "grounded": grounded,
        "steps": state.get("steps", 0) + 1,
    }


# ---------- Graph builder ----------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node())
    g.add_node("after_tools", after_tools)

    g.set_entry_point("llm")

    # After LLM: either go run tools (if tool_calls) or finish.
    g.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "final": END})

    # After tools: mark grounded/step++ then go back to LLM (loop).
    g.add_edge("tools", "after_tools")
    g.add_edge("after_tools", "llm")

    return g.compile()
