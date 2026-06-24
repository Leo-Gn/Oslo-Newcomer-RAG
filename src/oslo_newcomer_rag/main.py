import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import ExitStack, closing
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.chat_flow import build_boundary_answer, infer_answer_language, retrieve_for_chat
from oslo_newcomer_rag.db.models import AnonymousFeedback, Document, DocumentChunk, Source
from oslo_newcomer_rag.db.session import check_database, create_engine_from_settings
from oslo_newcomer_rag.generation import (
    ChatConfigError,
    ChatMessage as GenerationChatMessage,
    ChatResponseError,
    DataCurrency,
    GroundedAnswer,
    OpenAICompatibleChatClient,
    build_chat_plan,
    build_general_chat_answer,
    build_grounded_answer,
)
from oslo_newcomer_rag.retrieval import (
    EmbeddingConfigError,
    OpenAICompatibleEmbeddingClient,
)
from oslo_newcomer_rag.sources import load_source_registry


RATE_LIMITED_PATHS = {
    "/api/chat": "chat_rate_limit_per_minute",
    "/api/feedback": "feedback_rate_limit_per_minute",
}
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_RATE_LIMIT_BUCKETS = 10_000


class ComponentHealth(BaseModel):
    status: str
    checked: bool


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    database: ComponentHealth


class StoredSource(BaseModel):
    owner: str
    url: str
    language: str
    category: str
    intended_coverage: dict
    collected_at: datetime | None
    official_last_updated_at: datetime | None
    chunk_count: int


class SourceSnapshotResponse(BaseModel):
    database_configured: bool
    total_sources: int
    total_chunks: int
    sources: list[StoredSource]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    ui_language: Literal["en", "no"] = "en"
    session_history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)


class ChatCitation(BaseModel):
    citation_id: str
    chunk_id: str
    source_owner: str
    source_url: str
    section_url: str
    section_heading: str
    collected_at: datetime
    official_last_updated_at: datetime | None


class ChatDataCurrency(BaseModel):
    collected_at: datetime | None
    official_last_updated_at: datetime | None


class ChatResponse(BaseModel):
    answer_id: str
    answer: str
    refused: bool
    disclaimer: str | None
    citations: list[ChatCitation]
    data_currency: ChatDataCurrency


class FeedbackRequest(BaseModel):
    answer_id: UUID
    rating: Literal[-1, 0, 1]
    citation_chunk_ids: list[UUID] = Field(default_factory=list, max_length=20)


class FeedbackResponse(BaseModel):
    feedback_id: UUID | None
    created_at: datetime | None
    cleared: bool = False


class RateLimiter:
    def __init__(
        self,
        *,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window_seconds = window_seconds
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, *, limit: int) -> bool:
        now = self.clock()
        cutoff = now - self.window_seconds
        if len(self._hits) > MAX_RATE_LIMIT_BUCKETS:
            self._prune(cutoff)
        if len(self._hits) > MAX_RATE_LIMIT_BUCKETS and key not in self._hits:
            return False
        hits = self._hits[key]
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True

    def _prune(self, cutoff: float) -> None:
        stale_keys = [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for key in stale_keys:
            self._hits.pop(key, None)


def package_version() -> str:
    try:
        return version("oslo-newcomer-rag")
    except PackageNotFoundError:
        return "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    api = FastAPI(
        title=app_settings.app_name,
        version=package_version(),
        docs_url=None if app_settings.app_env == "production" else "/docs",
        redoc_url=None if app_settings.app_env == "production" else "/redoc",
        openapi_url=None if app_settings.app_env == "production" else "/openapi.json",
    )
    api.state.settings = app_settings
    api.state.rate_limiter = RateLimiter()

    @api.middleware("http")
    async def security_middleware(request: Request, call_next) -> Response:
        checked_request, early_response = await _preflight_security_response(
            request,
            app_settings,
            api.state.rate_limiter,
        )
        if early_response:
            _set_security_headers(early_response, app_settings)
            return early_response

        response = await call_next(checked_request)
        _set_security_headers(response, app_settings)
        return response

    @api.get("/healthz", response_model=HealthResponse, tags=["health"])
    async def healthz() -> HealthResponse:
        db_status = "not_configured"
        db_checked = False
        app_status = "ok"

        if app_settings.has_database_config:
            db_checked = True
            try:
                db_status = "ok" if check_database(app_settings) else "unreachable"
            except Exception:
                db_status = "unreachable"
            if db_status != "ok":
                app_status = "degraded"

        return HealthResponse(
            status=app_status,
            app=app_settings.app_name,
            version=package_version(),
            environment=app_settings.app_env,
            database=ComponentHealth(status=db_status, checked=db_checked),
        )

    @api.get("/api/sources", response_model=SourceSnapshotResponse, tags=["sources"])
    async def sources() -> SourceSnapshotResponse:
        if not app_settings.has_database_config:
            return SourceSnapshotResponse(
                database_configured=False,
                total_sources=0,
                total_chunks=0,
                sources=[],
            )

        engine = create_engine_from_settings(app_settings)
        try:
            with Session(engine) as session:
                registry_urls = [source.url for source in load_source_registry().sources]
                stored_sources = session.scalars(
                    select(Source)
                    .where(Source.url.in_(registry_urls))
                    .order_by(Source.owner, Source.category, Source.url)
                ).all()
                source_rows = [_stored_source_response(session, source) for source in stored_sources]
                total_chunks = (
                    session.scalar(
                        select(func.count(DocumentChunk.id))
                        .join(Source, Source.id == DocumentChunk.source_id)
                        .where(Source.url.in_(registry_urls))
                    )
                    or 0
                )
        finally:
            engine.dispose()

        return SourceSnapshotResponse(
            database_configured=True,
            total_sources=len(source_rows),
            total_chunks=total_chunks,
            sources=source_rows,
        )

    @api.post("/api/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest) -> ChatResponse:
        session_history = [
            GenerationChatMessage(role=message.role, content=message.content)
            for message in request.session_history
        ]
        answer_language = infer_answer_language(request.question, request.ui_language, session_history)

        direct_answer = build_boundary_answer(request.question, answer_language)
        if direct_answer:
            return _chat_response(direct_answer)

        if not app_settings.has_llm_config and not app_settings.has_database_config:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

        with ExitStack() as resources:
            try:
                chat_client = resources.enter_context(closing(OpenAICompatibleChatClient(app_settings)))
                chat_plan = build_chat_plan(
                    question=request.question,
                    ui_language=answer_language,
                    chat_client=chat_client,
                    session_history=session_history,
                )

                if chat_plan.mode == "general_chat":
                    answer = build_general_chat_answer(
                        question=request.question,
                        ui_language=answer_language,
                        chat_client=chat_client,
                        session_history=session_history,
                    )
                    return _chat_response(answer)

                if not app_settings.has_database_config:
                    raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

                engine = create_engine_from_settings(app_settings)
                resources.callback(engine.dispose)
                embedder = resources.enter_context(closing(OpenAICompatibleEmbeddingClient(app_settings)))
                with Session(engine) as session:
                    retrieval = retrieve_for_chat(
                        session,
                        embedder,
                        question=request.question,
                        ui_language=answer_language,
                        session_history=session_history,
                        planned_query=chat_plan.retrieval_query,
                    )
                    answer = build_grounded_answer(
                        question=request.question,
                        ui_language=answer_language,
                        retrieval=retrieval,
                        chat_client=chat_client,
                        session_history=session_history,
                    )
                return _chat_response(answer)
            except (ChatConfigError, EmbeddingConfigError, ValueError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ChatResponseError as exc:
                raise HTTPException(status_code=502, detail="Model provider returned an invalid answer") from exc
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail="Model provider request failed") from exc

    @api.post(
        "/api/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["feedback"],
    )
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        if not app_settings.has_database_config:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

        engine = create_engine_from_settings(app_settings)
        try:
            with Session(engine) as session:
                session.execute(delete(AnonymousFeedback).where(AnonymousFeedback.answer_id == request.answer_id))
                if request.rating == 0:
                    session.commit()
                    return FeedbackResponse(feedback_id=None, created_at=None, cleared=True)

                row = AnonymousFeedback(
                    answer_id=request.answer_id,
                    rating=request.rating,
                    citation_chunk_ids=request.citation_chunk_ids,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
        finally:
            engine.dispose()

        return FeedbackResponse(feedback_id=row.id, created_at=row.created_at)

    mount_frontend(api)
    return api


def mount_frontend(api: FastAPI) -> None:
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (frontend_dist / "index.html").exists():
        api.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


app = create_app()


def _stored_source_response(session: Session, source: Source) -> StoredSource:
    latest_document = session.scalar(
        select(Document)
        .where(Document.source_id == source.id)
        .order_by(Document.collected_at.desc(), Document.id.desc())
        .limit(1)
    )
    chunk_count = session.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.source_id == source.id)
    )

    return StoredSource(
        owner=source.owner,
        url=source.url,
        language=source.language,
        category=source.category,
        intended_coverage=source.intended_coverage,
        collected_at=latest_document.collected_at if latest_document else None,
        official_last_updated_at=latest_document.official_last_updated_at if latest_document else None,
        chunk_count=chunk_count or 0,
    )


def _chat_response(answer: GroundedAnswer) -> ChatResponse:
    return ChatResponse(
        answer_id=answer.answer_id,
        answer=answer.answer,
        refused=answer.refused,
        disclaimer=answer.disclaimer,
        citations=[
            ChatCitation(
                citation_id=citation.citation_id,
                chunk_id=citation.chunk_id,
                source_owner=citation.source_owner,
                source_url=citation.source_url,
                section_url=citation.section_url,
                section_heading=citation.section_heading,
                collected_at=citation.collected_at,
                official_last_updated_at=citation.official_last_updated_at,
            )
            for citation in answer.citations
        ],
        data_currency=_chat_data_currency(answer.data_currency),
    )


def _chat_data_currency(data_currency: DataCurrency) -> ChatDataCurrency:
    return ChatDataCurrency(
        collected_at=data_currency.collected_at,
        official_last_updated_at=data_currency.official_last_updated_at,
    )


async def _preflight_security_response(
    request: Request,
    settings: Settings,
    rate_limiter: RateLimiter,
) -> tuple[Request, JSONResponse | None]:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return request, None

    body = await _read_limited_body(request, settings.request_body_limit_bytes)
    if body is None:
        return request, JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": "Request body is too large"},
        )
    request._body = body

    limit_name = RATE_LIMITED_PATHS.get(request.url.path)
    if not limit_name or not settings.rate_limit_enabled:
        return request, None

    limit = int(getattr(settings, limit_name))
    client_key = _client_rate_limit_key(request)
    if rate_limiter.allow(f"{request.url.path}:{client_key}", limit=limit):
        return request, None

    return request, JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests. Please wait a little before trying again."},
        headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS))},
    )


async def _read_limited_body(request: Request, limit: int) -> bytes | None:
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            if int(raw_length) > limit:
                return None
        except ValueError:
            return None

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


def _client_rate_limit_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _set_security_headers(response: Response, settings: Settings) -> None:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), payment=()",
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")

    if settings.app_env == "production":
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "base-uri 'self'; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'; "
                "img-src 'self' data:; "
                "object-src 'none'; "
                "script-src 'self'; "
                "style-src 'self'"
            ),
        )
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
