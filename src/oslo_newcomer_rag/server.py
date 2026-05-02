import uvicorn

from oslo_newcomer_rag.config import Settings


def run(settings: Settings) -> None:
    uvicorn.run(
        "oslo_newcomer_rag.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )

