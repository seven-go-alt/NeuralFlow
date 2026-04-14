from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.core.context import ContextBuilder
from app.core.intent_router import IntentDetectionResult, IntentPolicy, IntentRouter
from app.core.llm import LLMClient
from app.memory.working import WorkingMemory

settings = get_settings()
app = FastAPI(title=settings.app_name)
intent_router = IntentRouter()
llm_client = LLMClient()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    prompt: str
    reply: str


class IntentDetectRequest(BaseModel):
    message: str


class IntentPolicyResponse(BaseModel):
    memory_strategy: str
    skill_whitelist: list[str]


class IntentDetectResponse(BaseModel):
    intents: list[str]
    primary_intent: str
    used_fallback: bool
    policies: dict[str, IntentPolicyResponse]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/api/intent/detect", response_model=IntentDetectResponse)
async def detect_intent(request: IntentDetectRequest) -> IntentDetectResponse:
    result = await intent_router.detect(request.message)
    return _serialize_intent_result(result)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    working_memory = WorkingMemory(session_id=request.session_id)
    working_memory.add_message("user", request.message)

    routed = await intent_router.detect(request.message)
    context_builder = ContextBuilder(session_id=request.session_id, working_mem=working_memory)
    prompt = await context_builder.build_prompt(request.message, routed.primary_intent)
    reply = await llm_client.generate(prompt)

    working_memory.add_message("assistant", reply)

    return ChatResponse(
        session_id=request.session_id,
        intent=routed.primary_intent,
        prompt=prompt,
        reply=reply,
    )


def _serialize_intent_result(result: IntentDetectionResult) -> IntentDetectResponse:
    return IntentDetectResponse(
        intents=result.intents,
        primary_intent=result.primary_intent,
        used_fallback=result.used_fallback,
        policies={
            name: IntentPolicyResponse(
                memory_strategy=policy.memory_strategy,
                skill_whitelist=policy.skill_whitelist,
            )
            for name, policy in result.policies.items()
        },
    )
