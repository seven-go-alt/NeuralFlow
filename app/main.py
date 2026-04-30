from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ValidationError
import httpx

from app.api.streaming import StreamTaskRegistry, create_sse_response
from app.config import get_settings
from app.config_manager import ConfigManager
from app.core.context import ContextBuilder
from app.core.intent_router import IntentDetectionResult, IntentRouter
from app.core.llm import LLMClient, build_rule_based_fallback_reply
from app.memory.working import WorkingMemory
from app.middleware.telemetry import TelemetryMiddleware
from app.plugins.manager import PluginManager
from app.middleware.tenant_isolation import TenantIsolationMiddleware
from app.models import TenantContext
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition, skill_registry
from app.utils.observability import configure_structured_logging, create_observability
from app.agents.react import ReActAgent

logger = logging.getLogger(__name__)

settings = get_settings()
audit_log_path = os.getenv("NEURALFLOW_AUDIT_LOG_PATH", "/tmp/neuralflow_audit.log")
observability = create_observability()
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TenantIsolationMiddleware, default_tenant_id=settings.tenant_default_id)
app.add_middleware(TelemetryMiddleware, observability=observability)
configure_structured_logging(logger_name="neuralflow.request", audit_log_path=audit_log_path)
intent_router = IntentRouter()
llm_client = LLMClient()
config_manager = ConfigManager()
mcp_client = MCPClient()
plugin_manager = PluginManager.from_env()
stream_registry = StreamTaskRegistry()
mcp_logger = configure_structured_logging(logger_name="neuralflow.mcp", audit_log_path=audit_log_path)

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return RedirectResponse("/docs")


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


def _strip_provider_prefix(model: str) -> str:
    """openai/gpt-5.4 -> gpt-5.4"""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _ensure_openai_prefix(model: str) -> str:
    """gpt-5.4 -> openai/gpt-5.4（已是 openai/ 或其他 provider 前缀则不动）"""
    if "/" in model:
        return model
    return f"openai/{model}"


@app.get("/api/models")
async def list_models():
    """从配置的 LLM 中转站拉取可用模型列表"""
    api_base = settings.llm_api_base
    api_key = settings.llm_api_key or settings.openai_api_key
    current = settings.litellm_model
    current_display = _strip_provider_prefix(current)
    if not api_base:
        return {"models": [], "current_model": current_display, "error": "LLM_API_BASE not configured"}
    url = api_base.rstrip("/") + "/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("data", []) if isinstance(data, dict) else []
        model_ids = [m.get("id", "") for m in raw if isinstance(m, dict) and m.get("id")]
        return {"models": sorted(model_ids), "current_model": current_display}
    except Exception as exc:
        return {"models": [], "current_model": current_display, "error": str(exc)}


class ModelSwitchRequest(BaseModel):
    model: str


@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """运行时切换当前使用的模型（仅内存生效，重启后恢复 .env 配置）"""
    litellm_model = _ensure_openai_prefix(request.model)
    settings.litellm_model = litellm_model
    llm_client.model = litellm_model
    return {"message": f"Model switched to {litellm_model}", "model": request.model, "litellm_model": litellm_model}


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
    _emit_response_generated_hook(
        reply=reply,
        request=request,
        intent=payload["intent"],
        prompt=payload["prompt"],
        skill_results=payload["skill_results"],
        tenant_context=getattr(http_request.state, "tenant", None),
    )
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
        final_reply = "".join(reply_parts)
        _emit_response_generated_hook(
            reply=final_reply,
            request=request,
            intent=payload["intent"],
            prompt=payload["prompt"],
            skill_results=payload["skill_results"],
            tenant_context=getattr(http_request.state, "tenant", None),
        )
        payload["working_memory"].add_message("assistant", final_reply)

    return await create_sse_response(request.session_id, event_source, stream_registry)


@app.post("/chat/react")
async def chat_react(http_request: Request, request: ChatRequest):
    """
    自主 Agent 接口：支持多步思考与工具调用循环 (ReAct 范式)
    """
    http_request.state.session_id = request.session_id
    tenant_context = getattr(http_request.state, "tenant", None)
    tenant_id = tenant_context.tenant_id if tenant_context else settings.tenant_default_id
    
    # 1. 识别意图
    routed = await intent_router.detect(request.message)
    http_request.state.intent = routed.primary_intent
    primary_policy = routed.policies[routed.primary_intent]
    
    # 2. 获取允许使用的工具
    selected_skills = skill_registry.get_allowed_skills(primary_policy.skill_whitelist)
    
    # 3. 初始化并运行 ReAct Agent
    agent = ReActAgent(mcp_client=mcp_client)
    
    # 执行循环
    result = await agent.execute(
        query=request.message,
        skills=selected_skills,
        session_id=request.session_id,
        tenant_context=tenant_context
    )
    
    # 4. 记录结果并更新记忆（将最终回答存入短期记忆）
    if result["final_answer"]:
        try:
            from app.memory.working import WorkingMemory
            working_memory = WorkingMemory(session_id=request.session_id, tenant_id=tenant_id)
            working_memory.add_message("user", request.message)
            working_memory.add_message("assistant", result["final_answer"])
        except Exception as e:
            logger.warning(f"Failed to update memory in react mode: {e}")

    # 5. 触发插件 Hook
    _emit_response_generated_hook(
        reply=result["final_answer"],
        request=request,
        intent=routed.primary_intent,
        prompt="ReAct Loop (Multi-step)",
        skill_results=[{"skill": s["tool"], "result": s["observation"]} for s in result["steps"] if s.get("type") == "tool_call"],
        tenant_context=tenant_context,
    )

    return {
        "session_id": request.session_id,
        "intent": routed.primary_intent,
        "final_answer": result["final_answer"],
        "steps": result["steps"],
        "total_iterations": result["iterations"],
        "reflections": result.get("reflections", []),
    }


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
    await _discover_remote_tools(session_id=request.session_id, intent=routed.primary_intent)
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
        started_at = perf_counter()
        try:
            result = await mcp_client.call_tool(skill.tool_name, payload, read_only=skill.read_only)
        except TypeError as exc:
            if "unexpected keyword argument 'read_only'" not in str(exc):
                raise
            result = await mcp_client.call_tool(skill.tool_name, payload)
        except Exception as exc:
            result = {"error": str(exc)}
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        mcp_logger.info(
            "tool_called",
            extra={
                "session_id": session_id,
                "intent": intent,
                "tool_name": skill.tool_name,
                "skill_name": skill.name,
                "duration_ms": duration_ms,
                "read_only": skill.read_only,
            },
        )
        results.append({"skill": skill.name, "result": result})
    return results


async def _discover_remote_tools(session_id: str, intent: str) -> None:
    try:
        tools = await mcp_client.list_tools()
    except Exception as exc:
        mcp_logger.warning(
            "tool_discovery_failed",
            extra={"session_id": session_id, "intent": intent, "error": str(exc)},
        )
        return

    for tool in tools:
        mcp_logger.info(
            "tool_discovered",
            extra={
                "session_id": session_id,
                "intent": intent,
                "tool_name": tool.name,
                "description": tool.description,
                "read_only": tool.read_only,
            },
        )



def _emit_response_generated_hook(
    *,
    reply: str,
    request: ChatRequest,
    intent: str,
    prompt: str,
    skill_results: list[dict[str, dict]],
    tenant_context: TenantContext | None,
) -> None:
    plugin_manager.emit(
        "on_response_generated",
        {
            "session_id": request.session_id,
            "message": request.message,
            "reply": reply,
            "intent": intent,
            "prompt": prompt,
            "skill_results": skill_results,
            "tenant_id": tenant_context.tenant_id if tenant_context is not None else settings.tenant_default_id,
            "tenant_roles": tenant_context.roles if tenant_context is not None else [],
            "tenant_scope": tenant_context.scope if tenant_context is not None else [settings.tenant_default_id],
        },
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
