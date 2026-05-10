import json

from fastapi.testclient import TestClient

from oslo_newcomer_rag.config import Settings
from oslo_newcomer_rag.main import create_app


def test_healthz_reports_running_app() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Oslo Newcomer Assistant"
    assert body["environment"] == "test"
    assert body["database"] == {"status": "not_configured", "checked": False}


def test_production_app_does_not_expose_openapi_docs() -> None:
    app = create_app(
        Settings(
            app_env="production",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
            llm_base_url="https://provider.example/v1",
            llm_api_key="test-key",
            llm_model="test-chat",
            embedding_model="test-embedding",
            embedding_dim=1536,
        )
    )
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_healthz_reports_unreachable_database(monkeypatch) -> None:
    def fail_check(settings: Settings) -> bool:
        raise OSError("database unavailable")

    monkeypatch.setattr("oslo_newcomer_rag.main.check_database", fail_check)
    app = create_app(
        Settings(
            app_env="test",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
        )
    )
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == {"status": "unreachable", "checked": True}


def test_healthz_reports_reachable_database(monkeypatch) -> None:
    monkeypatch.setattr("oslo_newcomer_rag.main.check_database", lambda settings: True)
    app = create_app(
        Settings(
            app_env="test",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
        )
    )
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == {"status": "ok", "checked": True}


def test_sources_endpoint_does_not_require_live_fetching() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    response = client.get("/api/sources")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "database_configured": False,
        "total_sources": 0,
        "total_chunks": 0,
        "sources": [],
    }


def test_chat_endpoint_requires_configured_database() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "What should I do after moving to Oslo?", "ui_language": "en"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "DATABASE_URL is not configured"


def test_chat_endpoint_answers_greeting_without_database(monkeypatch) -> None:
    class FakeChatClient:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def complete(self, messages) -> str:
            if "Return only JSON with keys: mode, retrieval_query" in messages[0].content:
                return json.dumps({"mode": "general_chat", "retrieval_query": ""})
            return json.dumps(
                {
                    "answer": (
                        "Hei! Jeg kan hjelpe med enkle spørsmål og bruke offisielle kilder "
                        "når du spør konkret om å flytte til Oslo."
                    ),
                    "refusal": False,
                }
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr("oslo_newcomer_rag.main.OpenAICompatibleChatClient", FakeChatClient)
    app = create_app(
        Settings(
            app_env="test",
            llm_base_url="https://provider.example/v1",
            llm_api_key="test-key",
            llm_model="test-chat",
        )
    )
    client = TestClient(app)

    response = client.post("/api/chat", json={"question": "hei", "ui_language": "no"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["citations"] == []
    assert "offisielle kilder" in body["answer"]


def test_chat_endpoint_answers_simple_chat_without_database(monkeypatch) -> None:
    class FakeChatClient:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def complete(self, messages) -> str:
            if "Return only JSON with keys: mode, retrieval_query" in messages[0].content:
                return json.dumps({"mode": "general_chat", "retrieval_query": ""})
            return json.dumps({"answer": "I am ready to help with Oslo newcomer questions.", "refusal": False})

        def close(self) -> None:
            pass

    monkeypatch.setattr("oslo_newcomer_rag.main.OpenAICompatibleChatClient", FakeChatClient)
    app = create_app(
        Settings(
            app_env="test",
            llm_base_url="https://provider.example/v1",
            llm_api_key="test-key",
            llm_model="test-chat",
        )
    )
    client = TestClient(app)

    response = client.post("/api/chat", json={"question": "How are you?", "ui_language": "en"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["citations"] == []
    assert "ready to help" in body["answer"]
