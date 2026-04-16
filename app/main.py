from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.core.context import ContextBuilder
from app.core.intent_router import IntentDetectionResult, IntentPolicy, IntentRouter
from app.core.llm import LLMClient
from app.memory.working import WorkingMemory
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition, skill_registry

settings = get_settings()
app = FastAPI(title=settings.app_name)
intent_router = IntentRouter()
llm_client = LLMClient()
mcp_client = MCPClient()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SkillResponse(BaseModel):
    name: str
    description: str


class SkillsListResponse(BaseModel):
    skills: list[SkillResponse]


class SkillExecutionResponse(BaseModel):
    skill: str
    result: dict


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    prompt: str
    reply: str
    used_skills: list[str]
    skill_results: list[SkillExecutionResponse]


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


@app.get("/api/skills", response_model=SkillsListResponse)
async def list_skills() -> SkillsListResponse:
    return SkillsListResponse(
        skills=[SkillResponse(name=skill.name, description=skill.description) for skill in skill_registry.list_skills()]
    )


@app.post("/api/intent/detect", response_model=IntentDetectResponse)
async def detect_intent(request: IntentDetectRequest) -> IntentDetectResponse:
    result = await intent_router.detect(request.message)
    return _serialize_intent_result(result)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    working_memory = WorkingMemory(session_id=request.session_id)
    working_memory.add_message("user", request.message)

    routed = await intent_router.detect(request.message)
    primary_policy = routed.policies[routed.primary_intent]
    selected_skills = skill_registry.get_allowed_skills(primary_policy.skill_whitelist)
    skill_results = await _run_skills(
        skills=selected_skills,
        session_id=request.session_id,
        intent=routed.primary_intent,
        user_query=request.message,
    )

    context_builder = ContextBuilder(session_id=request.session_id, working_mem=working_memory)
    prompt = await context_builder.build_prompt(
        request.message,
        routed.primary_intent,
        memory_strategy=primary_policy.memory_strategy,
        skill_whitelist=primary_policy.skill_whitelist,
        skill_results=skill_results,
    )
    reply = await llm_client.generate(prompt)

    working_memory.add_message("assistant", reply)

    return ChatResponse(
        session_id=request.session_id,
        intent=routed.primary_intent,
        prompt=prompt,
        reply=reply,
        used_skills=[item["skill"] for item in skill_results],
        skill_results=[SkillExecutionResponse(**item) for item in skill_results],
    )


async def _run_skills(
    skills: list[SkillDefinition],
    session_id: str,
    intent: str,
    user_query: str,
) -> list[dict[str, dict]]:
    results: list[dict[str, dict]] = []
    for skill in skills:
        payload = {
            "session_id": session_id,
            "intent": intent,
            "input": user_query,
        }
        result = await mcp_client.call_tool(skill.tool_name, payload)
        results.append({"skill": skill.name, "result": result})
    return results


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
