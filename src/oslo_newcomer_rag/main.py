from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from pydantic import BaseModel

from oslo_newcomer_rag.config import Settings, get_settings


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
        db_status = "configured" if app_settings.has_database_config else "not_configured"
        return HealthResponse(
            status="ok",
            app=app_settings.app_name,
            version=package_version(),
            environment=app_settings.app_env,
            database=ComponentHealth(status=db_status, checked=False),
        )

    return api


app = create_app()

