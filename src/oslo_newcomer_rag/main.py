from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Literal
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.chat_flow import build_direct_answer, retrieve_for_chat
from oslo_newcomer_rag.db.models import AnonymousFeedback, Document, DocumentChunk, Source
from oslo_newcomer_rag.db.session import check_database, create_engine_from_settings
from oslo_newcomer_rag.generation import (
    ChatConfigError,
    ChatMessage as GenerationChatMessage,
    ChatResponseError,
    DataCurrency,
    GroundedAnswer,
    OpenAICompatibleChatClient,
    build_grounded_answer,
)
from oslo_newcomer_rag.retrieval import (
    EmbeddingConfigError,
    OpenAICompatibleEmbeddingClient,
)
from oslo_newcomer_rag.sources import load_source_registry


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
        docs_url="/docs",
        redoc_url="/redoc",
    )
    api.state.settings = app_settings

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
        direct_answer = build_direct_answer(request.question, request.ui_language)
        if direct_answer:
            return _chat_response(direct_answer)

        if not app_settings.has_database_config:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

        engine = create_engine_from_settings(app_settings)
        embedder: OpenAICompatibleEmbeddingClient | None = None
        chat_client: OpenAICompatibleChatClient | None = None
        try:
            embedder = OpenAICompatibleEmbeddingClient(app_settings)
            chat_client = OpenAICompatibleChatClient(app_settings)
            session_history = [
                GenerationChatMessage(role=message.role, content=message.content)
                for message in request.session_history
            ]
            with Session(engine) as session:
                retrieval = retrieve_for_chat(
                    session,
                    embedder,
                    question=request.question,
                    ui_language=request.ui_language,
                    session_history=session_history,
                )
                answer = build_grounded_answer(
                    question=request.question,
                    ui_language=request.ui_language,
                    retrieval=retrieval,
                    chat_client=chat_client,
                    session_history=session_history,
                )
        except (ChatConfigError, EmbeddingConfigError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ChatResponseError as exc:
            raise HTTPException(status_code=502, detail="Model provider returned an invalid answer") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="Model provider request failed") from exc
        finally:
            if embedder:
                embedder.close()
            if chat_client:
                chat_client.close()
            engine.dispose()

        return _chat_response(answer)

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

    return api


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
