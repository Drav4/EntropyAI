import os, uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware
from ..config import UPLOAD_DIR
from ..models import ChatRequest
from ..utils.attachments import extract_attachment_candidates
from ..agent.graph import build_graph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

graph = build_graph()

def create_app() -> FastAPI:
    app = FastAPI(title="LangGraph ML Assistant", version="4.0.0")

    # ✅ CORS here so it’s guaranteed to wrap *this* app instance
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[  # list all dev origins you use
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,      # set False if you don't send cookies/auth
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            saved.append({"name": f.filename, "id": fid, "url": f"/files/{fid}",
                          "size": len(data), "type": f.content_type or ""})
        return saved

    @app.post("/chat")
    def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="Empty 'messages' array")

        # Build initial state
        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        attachments = extract_attachment_candidates(last_user.content) if last_user else []

        msgs = []
        for m in req.messages:
            if m.role == "user":
                msgs.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                msgs.append(AIMessage(content=m.content))
            elif m.role == "system":
                msgs.append(SystemMessage(content=m.content))

        state = {"messages": msgs, "attachments": attachments, "grounded": False, "steps": 0}
        final = graph.invoke(state)
        reply_msg = final["messages"][-1]
        # If the last message is a tool message (shouldn't be), back off to previous AI message
        if hasattr(reply_msg, "name"):
            # find last AI content
            for m in reversed(final["messages"]):
                if isinstance(m, AIMessage):
                    reply_msg = m; break

        return {"reply": reply_msg.content}

    @app.post("/debug/echo")
    async def echo(req: Request):
        return {"you_sent": await req.json()}

    return app
