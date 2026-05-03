from importlib.metadata import PackageNotFoundError, version
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.db.models import Document, DocumentChunk, Source
from oslo_newcomer_rag.db.session import create_engine_from_settings
from oslo_newcomer_rag.db.session import check_database
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
