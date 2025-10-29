# server/agent/graph.py
from langgraph.graph import StateGraph
from .state import GraphState
from .nodes.agent_node import agent_node
from .nodes.tool_node import tool_node
from .nodes.writer_node import writer_node

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("agent", agent_node)
    g.add_node("tool", tool_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("agent")

    def next_after_agent(state: GraphState) -> str:
        if state.tool_calls:
            return "tool"
        # No tool_calls → always go to writer (agent never produces final prose)
        return "writer"

    g.add_conditional_edges("agent", next_after_agent)
    g.add_edge("tool", "agent")   # loop until agent says READY_FOR_WRITER
    # writer is terminal
    return g.compile()
