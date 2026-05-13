import pytest
from pydantic import ValidationError

from oslo_newcomer_rag.config import Settings, get_settings


def test_development_settings_do_not_require_secrets() -> None:
    settings = Settings(app_env="development")

    assert settings.app_env == "development"
    assert settings.llm_api_key is None


def test_production_settings_require_provider_and_database_config() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(app_env="production")

    message = str(exc_info.value)
    assert "DATABASE_URL" in message
    assert "LLM_API_KEY" in message
    assert "EMBEDDING_DIM" in message


def test_production_settings_accept_required_values() -> None:
    settings = Settings(
        app_env="production",
        database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
        llm_base_url="https://api.example.com/v1",
        llm_api_key="test-key",
        llm_model="chat-model",
        embedding_model="embedding-model",
        embedding_dim=1536,
    )

    assert settings.has_database_config is True


def test_production_llm_base_url_must_use_https() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            app_env="production",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
            llm_base_url="http://provider.example/v1",
            llm_api_key="test-key",
            llm_model="chat-model",
            embedding_model="embedding-model",
            embedding_dim=1536,
        )

    assert "LLM_BASE_URL must use https in production" in str(exc_info.value)


def test_llm_base_url_must_not_contain_credentials() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            llm_base_url="https://user:pass@provider.example/v1",
            llm_api_key="test-key",
            llm_model="chat-model",
        )

    assert "LLM_BASE_URL must not contain credentials" in str(exc_info.value)


def test_app_settings_load_local_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
                "LLM_BASE_URL=https://api.example.com/v1",
                "LLM_API_KEY=test-key",
                "LLM_MODEL=test-chat",
                "EMBEDDING_MODEL=test-embedding",
                "EMBEDDING_DIM=1536",
            ]
        )
    )

    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()

    assert settings.database_url == "postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer"
    assert settings.has_database_config is True
