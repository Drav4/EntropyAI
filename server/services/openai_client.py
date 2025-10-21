from langchain_openai import ChatOpenAI
from ..config import OPENAI_API_KEY, OPENAI_MODEL
from fastapi import HTTPException

def make_llm():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set.")
    # Deterministic config
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.0,
        top_p=1.0,
        timeout=90,
    )
