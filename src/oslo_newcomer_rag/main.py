from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from pydantic import BaseModel

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.db.session import check_database


class ComponentHealth(BaseModel):
    status: str
    checked: bool


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    database: ComponentHealth


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

    return api


app = create_app()
