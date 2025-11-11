# server/agent/graph.py
from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes.guard_node import guard_node, REFUSAL   # ← import REFUSAL
from .nodes.agent_node import agent_node
from .nodes.tool_node import tool_node
from .nodes.writer_node import writer_node

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("guard", guard_node)     # ← NEW
    g.add_node("agent", agent_node)
    g.add_node("tool", tool_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("guard")          # ← start at guard

    # After guard: end if it already wrote the refusal, else continue to agent
    def route_after_guard(state: GraphState):
        try:
            # Prefer explicit check if guard set final_answer
            if getattr(state, "final_answer", "") == REFUSAL:
                return END
        except Exception:
            pass
        # Fallback: check last AI message equals REFUSAL
        if state.messages and str(getattr(state.messages[-1], "content", "")) == REFUSAL:
            return END
        return "agent"

    g.add_conditional_edges("guard", route_after_guard)

    # Normal DS flow (unchanged)
    def next_after_agent(state: GraphState) -> str:
        if state.tool_calls:
            return "tool"
        return "writer"

    g.add_conditional_edges("agent", next_after_agent)
    g.add_edge("tool", "agent")  # loop until no tool_calls
    # writer → END implicitly
    return g.compile()
