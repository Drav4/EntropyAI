from typing import List, Literal
from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]

class ChatTurn(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatTurn]
