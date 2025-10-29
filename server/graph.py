from typing import Dict, Any
import pandas as pd
from .nodes.llm import BaseLLMClient, make_llm
from .core.toolsim import ToolBoundShim
from .tools.registry import TOOLS
from .nodes.agent import run_agentic_loop
from .nodes.write_answer import write_answer
from .core.filestore import csv_path

# Optional helpers: bootstrap minimal evidence if model didn’t call tools
def _ensure_minimum_evidence(state: Dict[str, Any], tb: ToolBoundShim):
    if state.get("tool_results"):
        return
    files = state.get("files") or []
    if not files:
        return
    fid = files[0].get("file_id") or files[0].get("id") or files[0].get("name")
    if not fid:
        return
    tb.invoke_tool({"name":"compute_dataset_facts","args":{"file_id_or_name":fid}}, state)

def run_pipeline(question: str, file_id: str | None = None, client: BaseLLMClient | None = None) -> Dict[str, Any]:
    client = client or make_llm()
    state: Dict[str, Any] = {
        "user_question": question,
        "files": [{"file_id": file_id}] if file_id else [],
        "tool_calls": [],
        "tool_results": [],
        "plots": [],
        "notes": [],
        "seed": 7,
    }

    tb = ToolBoundShim(client=client, tools=TOOLS)

    # 1) Agent plans & acts (decides tool calls; ToolBoundShim logs evidence)
    _ = run_agentic_loop(client, tb, question, state, max_turns=6)

    # 2) Ensure there is at least minimal evidence (facts) if no tool was called
    _ensure_minimum_evidence(state, tb)

    # 3) Writer produces final Markdown answer with technical facts
    reply_text = write_answer(client, state)

    # You may still want to return evidence for UI drill-down
    return {"reply": reply_text, "reproducibility": {
        "files": state.get("files", []),
        "tool_calls": state.get("tool_calls", []),
        "results": state.get("tool_results", []),
        "plots": state.get("plots", []),
        "notes": state.get("notes", []),
    }}
