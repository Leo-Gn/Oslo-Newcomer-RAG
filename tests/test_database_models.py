from sqlalchemy import CheckConstraint, Computed, UniqueConstraint

from oslo_newcomer_rag.db.models import AnonymousFeedback, Base, DocumentChunk, Embedding


def test_database_metadata_contains_step_three_tables() -> None:
    assert set(Base.metadata.tables) == {
        "anonymous_feedback",
        "document_chunks",
        "documents",
        "embeddings",
        "retrieval_logs",
        "sources",
    }


def test_chunks_have_generated_search_vector() -> None:
    search_vector = DocumentChunk.__table__.c.search_vector

    assert isinstance(search_vector.computed, Computed)
    assert "to_tsvector" in str(search_vector.computed.sqltext)


def test_embeddings_are_one_per_chunk_and_dimension_checked() -> None:
    constraints = Embedding.__table__.constraints

    assert Embedding.__table__.c.chunk_id.unique is True
    assert any(
        isinstance(constraint, CheckConstraint)
        and constraint.name == "ck_embeddings_dim_positive"
        for constraint in constraints
    )


def test_feedback_schema_stays_anonymous() -> None:
    columns = set(AnonymousFeedback.__table__.c.keys())
    constraints = AnonymousFeedback.__table__.constraints

    assert columns == {"id", "answer_id", "rating", "citation_chunk_ids", "created_at"}
    assert not {"question", "answer", "chat_text"} & columns
    assert any(
        isinstance(constraint, CheckConstraint)
        and constraint.name == "ck_anonymous_feedback_rating"
        for constraint in constraints
    )


def test_documents_and_chunks_have_idempotency_constraints() -> None:
    constraints = Base.metadata.tables["documents"].constraints | Base.metadata.tables["document_chunks"].constraints

    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_documents_source_hash"
        for constraint in constraints
    )
    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_document_chunks_order"
        for constraint in constraints
    )
