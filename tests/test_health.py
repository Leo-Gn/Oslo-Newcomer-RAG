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


def test_healthz_reports_database_configuration_without_connecting() -> None:
    app = create_app(
        Settings(
            app_env="test",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
        )
    )
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["database"] == {"status": "configured", "checked": False}
