# server/agent/state.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class GraphState(BaseModel):
    messages: List[BaseMessage]
    evidence: Dict[str, Any] = Field(default_factory=dict)   # raw tool payloads (kept internal)
    facts: Dict[str, Any] = Field(default_factory=dict)       # parsed, tool-agnostic facts
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    final_answer: Optional[str] = None
    steps: int = 0
    max_steps: int = 8
