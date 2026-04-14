from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.core.context import ContextBuilder
from app.core.llm import LLMClient
from app.core.router import IntentRouter
from app.memory.working import WorkingMemory

settings = get_settings()
app = FastAPI(title=settings.app_name)
router = IntentRouter()
llm_client = LLMClient()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    prompt: str
    reply: str


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    working_memory = WorkingMemory(session_id=request.session_id)
    working_memory.add_message("user", request.message)

    routed = router.route(request.message)
    context_builder = ContextBuilder(session_id=request.session_id, working_mem=working_memory)
    prompt = await context_builder.build_prompt(request.message, routed.name)
    reply = await llm_client.generate(prompt)

    working_memory.add_message("assistant", reply)

    return ChatResponse(
        session_id=request.session_id,
        intent=routed.name,
        prompt=prompt,
        reply=reply,
    )
