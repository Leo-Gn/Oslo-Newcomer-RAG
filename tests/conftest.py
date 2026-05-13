import pytest


PROJECT_ENV_VARS = (
    "APP_ENV",
    "APP_HOST",
    "APP_PORT",
    "DATABASE_URL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "REQUEST_BODY_LIMIT_BYTES",
    "RATE_LIMIT_ENABLED",
    "CHAT_RATE_LIMIT_PER_MINUTE",
    "FEEDBACK_RATE_LIMIT_PER_MINUTE",
)


@pytest.fixture(autouse=True)
def clear_project_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROJECT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
