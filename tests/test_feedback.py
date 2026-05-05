from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from oslo_newcomer_rag.config import Settings
from oslo_newcomer_rag.main import create_app


def test_feedback_endpoint_requires_configured_database() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    response = client.post(
        "/api/feedback",
        json={
            "answer_id": str(uuid4()),
            "rating": 1,
            "citation_chunk_ids": [str(uuid4())],
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "DATABASE_URL is not configured"


def test_feedback_endpoint_stores_metadata_only(monkeypatch) -> None:
    stored_rows = []
    deleted_rows = []

    class FakeEngine:
        def dispose(self) -> None:
            pass

    class FakeSession:
        def __init__(self, engine: FakeEngine) -> None:
            self.engine = engine

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def add(self, row) -> None:
            row.id = uuid4()
            row.created_at = datetime(2026, 5, 5, 12, 0, tzinfo=UTC)
            stored_rows.append(row)

        def execute(self, statement) -> None:
            deleted_rows.append(statement)

        def commit(self) -> None:
            pass

        def refresh(self, row) -> None:
            pass

    monkeypatch.setattr("oslo_newcomer_rag.main.create_engine_from_settings", lambda settings: FakeEngine())
    monkeypatch.setattr("oslo_newcomer_rag.main.Session", FakeSession)

    answer_id = uuid4()
    citation_id = uuid4()
    app = create_app(
        Settings(
            app_env="test",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/feedback",
        json={
            "answer_id": str(answer_id),
            "rating": -1,
            "citation_chunk_ids": [str(citation_id)],
            "question": "this must not be stored",
            "answer": "this must not be stored",
        },
    )

    assert response.status_code == 201
    assert UUID(response.json()["feedback_id"])
    assert len(stored_rows) == 1
    row = stored_rows[0]
    assert row.answer_id == answer_id
    assert row.rating == -1
    assert row.citation_chunk_ids == [citation_id]
    assert "question" not in row.__dict__
    assert "answer" not in row.__dict__
    assert len(deleted_rows) == 1


def test_feedback_endpoint_clears_existing_feedback(monkeypatch) -> None:
    stored_rows = []
    deleted_rows = []

    class FakeEngine:
        def dispose(self) -> None:
            pass

    class FakeSession:
        def __init__(self, engine: FakeEngine) -> None:
            self.engine = engine

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def add(self, row) -> None:
            stored_rows.append(row)

        def execute(self, statement) -> None:
            deleted_rows.append(statement)

        def commit(self) -> None:
            pass

        def refresh(self, row) -> None:
            pass

    monkeypatch.setattr("oslo_newcomer_rag.main.create_engine_from_settings", lambda settings: FakeEngine())
    monkeypatch.setattr("oslo_newcomer_rag.main.Session", FakeSession)

    app = create_app(
        Settings(
            app_env="test",
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/feedback",
        json={
            "answer_id": str(uuid4()),
            "rating": 0,
            "citation_chunk_ids": [],
        },
    )

    assert response.status_code == 201
    assert response.json() == {"feedback_id": None, "created_at": None, "cleared": True}
    assert stored_rows == []
    assert len(deleted_rows) == 1


def test_feedback_endpoint_rejects_invalid_rating() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    response = client.post(
        "/api/feedback",
        json={
            "answer_id": str(uuid4()),
            "rating": 2,
            "citation_chunk_ids": [],
        },
    )

    assert response.status_code == 422
