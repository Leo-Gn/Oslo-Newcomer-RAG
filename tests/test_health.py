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
