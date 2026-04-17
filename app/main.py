from __future__ import annotations

import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError

from app.api.streaming import StreamTaskRegistry, create_sse_response
from app.config import get_settings
from app.config_manager import ConfigManager
from app.core.context import ContextBuilder
from app.core.intent_router import IntentDetectionResult, IntentRouter
from app.core.llm import LLMClient, build_rule_based_fallback_reply
from app.memory.working import WorkingMemory
from app.middleware.telemetry import TelemetryMiddleware
from app.middleware.tenant_isolation import TenantIsolationMiddleware
from app.models import TenantContext
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition, skill_registry
from app.utils.observability import configure_structured_logging, create_observability

settings = get_settings()
audit_log_path = os.getenv("NEURALFLOW_AUDIT_LOG_PATH", "/tmp/neuralflow_audit.log")
observability = create_observability()
app = FastAPI(title=settings.app_name)
app.add_middleware(TenantIsolationMiddleware, default_tenant_id=settings.tenant_default_id)
app.add_middleware(TelemetryMiddleware, observability=observability)
configure_structured_logging(logger_name="neuralflow.request", audit_log_path=audit_log_path)
intent_router = IntentRouter()
llm_client = LLMClient()
config_manager = ConfigManager()
mcp_client = MCPClient()
stream_registry = StreamTaskRegistry()


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


class AdminConfigResponse(BaseModel):
    config: dict
    audit_entry: dict | None = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/metrics")
async def metrics():
    return observability.metrics_response()


@app.get("/admin/config", response_model=AdminConfigResponse)
async def get_runtime_config(http_request: Request) -> AdminConfigResponse:
    _verify_admin_secret(http_request)
    snapshot = await config_manager.get_snapshot()
    return AdminConfigResponse(config=snapshot.model_dump())


@app.patch("/admin/config", response_model=AdminConfigResponse)
async def patch_runtime_config(http_request: Request, patch: dict) -> AdminConfigResponse:
    _verify_admin_secret(http_request)
    source_ip = _get_client_ip(http_request)
    try:
        updated = await config_manager.update(patch, source_ip=source_ip, actor="admin_api")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc
    audit_entries = await config_manager.list_audit_entries()
    latest_audit = audit_entries[-1].model_dump(mode="json") if audit_entries else None
    return AdminConfigResponse(config=updated.model_dump(), audit_entry=latest_audit)


@app.get("/api/skills", response_model=SkillsListResponse)
async def list_skills() -> SkillsListResponse:
    return SkillsListResponse(
        skills=[SkillResponse(name=skill.name, description=skill.description) for skill in skill_registry.list_skills()]
    )


@app.post("/api/intent/detect", response_model=IntentDetectResponse)
async def detect_intent(http_request: Request, request: IntentDetectRequest) -> IntentDetectResponse:
    result = await intent_router.detect(request.message)
    http_request.state.intent = result.primary_intent
    return _serialize_intent_result(result)


@app.post("/chat", response_model=ChatResponse)
async def chat(http_request: Request, request: ChatRequest) -> ChatResponse:
    http_request.state.session_id = request.session_id
    payload = await _prepare_chat(request, tenant_context=getattr(http_request.state, "tenant", None))
    http_request.state.intent = payload["intent"]
    reply = await _generate_reply_with_fallback(payload["prompt"])
    payload["working_memory"].add_message("assistant", reply)
    return ChatResponse(
        session_id=request.session_id,
        intent=payload["intent"],
        prompt=payload["prompt"],
        reply=reply,
        used_skills=[item["skill"] for item in payload["skill_results"]],
        skill_results=[SkillExecutionResponse(**item) for item in payload["skill_results"]],
    )


@app.post("/chat/stream")
async def chat_stream(http_request: Request, request: ChatRequest, include_thinking: bool | None = None):
    http_request.state.session_id = request.session_id
    payload = await _prepare_chat(request, tenant_context=getattr(http_request.state, "tenant", None))
    http_request.state.intent = payload["intent"]
    runtime_config = await config_manager.get_snapshot()
    include_reasoning = runtime_config.stream_thinking_enabled if include_thinking is None else include_thinking

    async def event_source() -> AsyncIterator[dict[str, dict | str | float]]:
        reply_parts: list[str] = []
        async for chunk in _stream_reply_with_fallback(payload["prompt"], include_thinking=include_reasoning):
            if chunk["event"] == "message":
                reply_parts.append(chunk["data"])
            yield {"event": chunk["event"], "data": {"delta": chunk["data"]}}
        payload["working_memory"].add_message("assistant", "".join(reply_parts))

    return await create_sse_response(request.session_id, event_source, stream_registry)


async def _prepare_chat(request: ChatRequest, tenant_context: TenantContext | None = None) -> dict:
    tenant_id = tenant_context.tenant_id if tenant_context is not None else settings.tenant_default_id
    try:
        working_memory = WorkingMemory(session_id=request.session_id, tenant_id=tenant_id)
    except TypeError as exc:
        if "unexpected keyword argument 'tenant_id'" not in str(exc):
            raise
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
        tenant_context=tenant_context,
    )

    try:
        context_builder = ContextBuilder(
            session_id=request.session_id,
            working_mem=working_memory,
            tenant_id=tenant_id,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'tenant_id'" not in str(exc):
            raise
        context_builder = ContextBuilder(session_id=request.session_id, working_mem=working_memory)
    prompt = await context_builder.build_prompt(
        request.message,
        routed.primary_intent,
        memory_strategy=primary_policy.memory_strategy,
        skill_whitelist=primary_policy.skill_whitelist,
        skill_results=skill_results,
    )
    return {
        "working_memory": working_memory,
        "intent": routed.primary_intent,
        "prompt": prompt,
        "skill_results": skill_results,
    }


async def _generate_reply_with_fallback(prompt: str) -> str:
    try:
        return await llm_client.generate(prompt)
    except Exception as exc:
        return build_rule_based_fallback_reply(prompt, error=exc)


async def _stream_reply_with_fallback(prompt: str, include_thinking: bool = False) -> AsyncIterator[dict[str, str]]:
    try:
        async for chunk in llm_client.stream_generate(prompt, include_thinking=include_thinking):
            yield chunk
    except Exception as exc:
        yield {"event": "message", "data": build_rule_based_fallback_reply(prompt, error=exc)}


async def _run_skills(
    skills: list[SkillDefinition],
    session_id: str,
    intent: str,
    user_query: str,
    tenant_context: TenantContext | None = None,
) -> list[dict[str, dict]]:
    results: list[dict[str, dict]] = []
    for skill in skills:
        payload = {
            "session_id": session_id,
            "intent": intent,
            "input": user_query,
        }
        if tenant_context is not None:
            payload.update(
                {
                    "tenant_id": tenant_context.tenant_id,
                    "tenant_roles": tenant_context.roles,
                    "tenant_scope": tenant_context.scope,
                }
            )
        try:
            result = await mcp_client.call_tool(skill.tool_name, payload, read_only=skill.read_only)
        except TypeError as exc:
            if "unexpected keyword argument 'read_only'" not in str(exc):
                raise
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


def _verify_admin_secret(request: Request) -> None:
    expected = os.getenv("ADMIN_SECRET_KEY")
    provided = _extract_admin_secret(request)
    if not expected or provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_admin_secret(request: Request) -> str | None:
    explicit_secret = request.headers.get("X-Admin-Secret")
    if explicit_secret:
        return explicit_secret

    authorization = request.headers.get("Authorization")
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _get_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"
