import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Source(TimestampMixin, Base):
    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    owner: Mapped[str] = mapped_column(String(120), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    intended_coverage: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source_id", "content_hash", name="uq_documents_source_hash"),
        Index("ix_documents_source_id", "source_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    official_last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text)

    source: Mapped["Source"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_order"),
        Index("ix_document_chunks_document_id", "document_id"),
        Index("ix_document_chunks_source_id", "source_id"),
        Index("ix_document_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_heading: Mapped[str] = mapped_column(Text, nullable=False)
    section_url: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(text, ''))", persisted=True),
        nullable=False,
    )
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    official_last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped["Document"] = relationship(back_populates="chunks")
    embedding: Mapped["Embedding | None"] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        CheckConstraint("embedding_dim > 0", name="ck_embeddings_dim_positive"),
        Index("ix_embeddings_chunk_id", "chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    embedding_model: Mapped[str] = mapped_column(String(160), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(Vector, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chunk: Mapped["DocumentChunk"] = relationship(back_populates="embedding")


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"
    __table_args__ = (
        Index("ix_retrieval_logs_created_at", "created_at"),
        Index("ix_retrieval_logs_query_hash", "query_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str | None] = mapped_column(String(12))
    filters: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    retrieved_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    ranking: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    low_confidence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class AnonymousFeedback(Base):
    __tablename__ = "anonymous_feedback"
    __table_args__ = (
        CheckConstraint("rating in (-1, 1)", name="ck_anonymous_feedback_rating"),
        Index("ix_anonymous_feedback_answer_id", "answer_id"),
        Index("ix_anonymous_feedback_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    answer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    citation_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
