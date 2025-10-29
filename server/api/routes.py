# server/api/routes.py
from __future__ import annotations
import os
import uuid
import re
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from server.config import UPLOAD_DIR
from server.models import ChatRequest
from server.utils.attachments import extract_attachment_candidates
from server.agent.graph import build_graph

# ---------------------------
# Compile graph once
# ---------------------------
graph = build_graph()

# ---------------------------
# Helpers
# ---------------------------

_TOOL_JSON_RE = re.compile(r'^\s*\{[\s\S]*"tool_name"\s*:\s*".+?"[\s\S]*\}\s*$', re.DOTALL)

def _as_lc_messages(raw_msgs: List[Dict[str, Any]]) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for m in raw_msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        elif role == "system":
            msgs.append(SystemMessage(content=content))
    return msgs

def _active_file_system_hint(attachments: List[Dict[str, Any]]) -> SystemMessage | None:
    if not attachments:
        return None
    # Use the ID (the filename stored under /files/<id>)
    fid = attachments[0].get("id") or attachments[0].get("name")
    if not fid:
        return None
    return SystemMessage(content=f"Active dataset id: '{fid}'")

# ---------------------------
# FastAPI app
# ---------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="LangGraph ML Assistant", version="5.0.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files for uploaded data
    app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

    @app.get("/")
    def root():
        return {"status": "ok", "chat": "/chat", "upload": "/upload", "files": "/files/<id>"}

    @app.post("/upload")
    async def upload(files: List[UploadFile] = File(...)):
        saved = []
        for f in files:
            _, ext = os.path.splitext(f.filename)
            fid = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join(UPLOAD_DIR, fid)
            data = await f.read()
            with open(path, "wb") as out:
                out.write(data)
            saved.append({
                "name": f.filename,
                "id": fid,
                "url": f"/files/{fid}",
                "size": len(data),
                "type": f.content_type or ""
            })
        return saved

    @app.post("/chat")
    def chat(req: ChatRequest):
        # 1) Validate
        if not req.messages:
            raise HTTPException(status_code=400, detail="Empty 'messages' array")

        # 2) Build LangChain messages
        msgs = _as_lc_messages([m.dict() if hasattr(m, "dict") else m for m in req.messages])

        # 3) Extract attachments from the last user message
        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        attachments = extract_attachment_candidates(last_user.content) if last_user else []

        # 4) Inject a system hint for Active dataset id (helps the agent pick the right file)
        hint = _active_file_system_hint(attachments)
        if hint:
            msgs.insert(0, hint)

        # 5) Initialize graph state and run
        state = {"messages": msgs, "evidence": {}, "facts": {}, "tool_calls": [], "final_answer": None, "steps": 0}
        final_state = graph.invoke(state)

        # 6) Choose reply from final_answer or last AI message
        reply: str = final_state.get("final_answer") or ""

        # Safety: if a raw tool JSON ever slipped through, hide it
        if _TOOL_JSON_RE.match(reply or ""):
            reply = "Processing analysis…"

        # Fallback to last AI message content if final_answer missing
        if not reply:
            for m in reversed(final_state.get("messages", [])):
                if isinstance(m, AIMessage) and (m.content or "").strip():
                    reply = m.content
                    break

        # Final guard: always return plain text in `reply`
        reply = (reply or "").strip()

        return {"reply": reply}

    @app.post("/debug/echo")
    async def echo(req: Request):
        return {"you_sent": await req.json()}

    return app
