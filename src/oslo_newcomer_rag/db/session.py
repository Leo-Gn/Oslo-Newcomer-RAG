from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from oslo_newcomer_rag.config import Settings


def create_engine_from_settings(settings: Settings) -> Engine:
    if not settings.database_url:
        raise ValueError("DATABASE_URL is not configured")

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 2},
    )


def check_database(settings: Settings) -> bool:
    if not settings.database_url:
        return False

    engine = create_engine_from_settings(settings)
    try:
        with engine.connect() as connection:
            connection.execute(text("select 1"))
    finally:
        engine.dispose()

    return True
