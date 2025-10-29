# server/llm.py
from __future__ import annotations
from typing import Optional, Union, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from server.config import OPENAI_API_KEY, OPENAI_MODEL

class SimpleChat:
    """
    Tiny adapter so our ToolBound can call .invoke(prompt_text: str) and get back an AIMessage.
    We inject a SystemMessage (if provided) correctly for the Chat Completions API.
    """
    def __init__(self, model: str, api_key: str, temperature: float = 0.2, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    def invoke(self, prompt_text: Union[str, List[BaseMessage]]) -> AIMessage:
        """
        Accepts either a plain string (our ToolBound passes a single string),
        or a list of BaseMessage (if you later want to call it that way).
        Always returns an AIMessage.
        """
        if isinstance(prompt_text, str):
            msgs: List[BaseMessage] = []
            if self.system_prompt:
                msgs.append(SystemMessage(content=self.system_prompt))
            msgs.append(HumanMessage(content=prompt_text))
        else:
            # Already a message list; prepend system if we have one and none exists yet
            msgs = list(prompt_text)
            if self.system_prompt and not any(isinstance(m, SystemMessage) for m in msgs):
                msgs = [SystemMessage(content=self.system_prompt)] + msgs

        out = self.llm.invoke(msgs)  # LangChain returns an AIMessage
        # Ensure we return an AIMessage (some wrappers may return Message-like objects)
        if isinstance(out, AIMessage):
            return out
        # Fallback: wrap content
        return AIMessage(content=getattr(out, "content", str(out)))

def make_llm(system_prompt: Optional[str] = None, *, temperature: float = 0.2) -> SimpleChat:
    """
    Factory used by agent and writer nodes.
    - No 'system' kwarg passed to the OpenAI client.
    - Properly injects SystemMessage at call time.
    - Returns an object with .invoke(...) that ToolBound can call with a single string.
    """
    if not OPENAI_API_KEY:
        # You can raise if you prefer:
        # raise RuntimeError("OPENAI_API_KEY is not set")
        # But returning a model that will error with a clear message is also fine.
        pass
    return SimpleChat(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        system_prompt=system_prompt,
    )
