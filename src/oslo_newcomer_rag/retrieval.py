from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import Select, delete, func, or_, select
from sqlalchemy.orm import Session

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.db.models import DocumentChunk, Embedding, RetrievalLog, Source
from oslo_newcomer_rag.db.session import create_engine_from_settings


VECTOR_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35
MIN_RETRIEVAL_SCORE = 0.28
MIN_VECTOR_SIMILARITY = 0.18
MIN_VECTOR_ONLY_SIMILARITY = 0.55
DEFAULT_BATCH_SIZE = 8
DEFAULT_CANDIDATE_LIMIT = 40
DEFAULT_RESULT_LIMIT = 6
MAX_EMBEDDING_RETRY_ATTEMPTS = 6
INITIAL_RETRY_SECONDS = 2.0
MAX_RETRY_SECONDS = 60.0
RETRIEVAL_GLOSSARY = {
    "skattekort": ("tax deduction card", "tax card"),
    "oppholdstillatelse": ("residence permit", "immigration permit"),
    "oppholdskort": ("residence card",),
    "statsborgerskap": ("citizenship", "Norwegian citizenship"),
    "familieinnvandring": ("family immigration",),
    "studentbolig": ("student housing",),
    "fastlege": ("general practitioner", "GP", "healthcare"),
    "d-nummer": ("D number", "identification number"),
    "d nummer": ("D number", "identification number"),
    "d-number": ("D number", "identification number"),
    "d number": ("D number", "identification number"),
    "fødselsnummer": ("national identity number", "identification number"),
    "personnummer": ("national identity number",),
    "eøs": ("EU EEA", "right of residence"),
    "eu/eøs": ("EU EEA", "right of residence"),
    "eea": ("EU EEA", "right of residence"),
    "arbeidstillatelse": ("work permit", "work immigration"),
    "permanent residency": ("permanent residence", "permanent residence permit"),
    "permanent opphold": ("permanent residence",),
    "citizenship": ("Norwegian citizenship", "Norwegian citizen", "UDI"),
    "norskkurs": ("Norwegian language course", "learn Norwegian"),
    "barnehage": ("kindergarten", "children families"),
    "flytter": ("moving to Norway", "first steps"),
    "skatt": ("tax",),
    "bolig": ("housing", "accommodation"),
    "studenter": ("students", "student services"),
    "student": ("student", "student services"),
    "helse": ("healthcare", "health services"),
    "helsetjenester": ("healthcare services",),
    "lære norsk": ("learn Norwegian", "Norwegian language course"),
    "faglært": ("skilled worker",),
    "faglærte": ("skilled workers",),
    "familier": ("children families",),
    "bestille time": ("book appointment", "appointment booking"),
    "appointment": ("book appointment", "SUA", "service centre foreign workers"),
    "book an appointment": ("book appointment", "SUA", "service centre foreign workers"),
    "sua": ("Service Centre for Foreign Workers", "book appointment"),
    "utenlandsk arbeidstaker": ("foreign worker", "service centre foreign workers"),
    "study permit": ("studies", "student residence permit", "UDI"),
    "studietillatelse": ("study permit", "student residence permit", "UDI"),
    "studenttillatelse": ("study permit", "student residence permit", "UDI"),
    "proof of funds": ("money", "funds", "student residence permit"),
    "penger": ("money", "funds", "proof of funds"),
    "deltid": ("part-time work", "work while studying"),
    "part-time": ("part-time work", "work while studying"),
    "waiting list": ("housing application", "student housing", "SiO"),
    "venteliste": ("waiting list", "housing application", "student housing"),
    "dagpenger": ("unemployment benefits", "jobseeker", "NAV"),
    "unemployment benefits": ("dagpenger", "jobseeker", "NAV"),
    "pet": ("rental", "housing", "accommodation"),
    "pets": ("rental", "housing", "accommodation"),
    "hund": ("pet", "rental", "housing"),
    "katt": ("pet", "rental", "housing"),
}


@dataclass(frozen=True)
class RetrievalFilters:
    language: str | None = None
    owners: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "owners": list(self.owners),
            "categories": list(self.categories),
            "source_urls": list(self.source_urls),
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    source_owner: str
    source_url: str
    category: str
    language: str
    section_heading: str
    section_url: str
    text: str
    collected_at: datetime
    official_last_updated_at: datetime | None
    score: float
    vector_score: float
    keyword_score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    low_confidence: bool
    language_fallback_used: bool = False


@dataclass(frozen=True)
class EmbeddingBatchResult:
    scanned_chunks: int
    embedded_chunks: int
    embedding_model: str
    embedding_dim: int


class EmbeddingConfigError(RuntimeError):
    pass


class EmbeddingResponseError(RuntimeError):
    pass


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.llm_base_url:
            raise EmbeddingConfigError("LLM_BASE_URL is not configured")
        if not settings.llm_api_key:
            raise EmbeddingConfigError("LLM_API_KEY is not configured")
        if not settings.embedding_model:
            raise EmbeddingConfigError("EMBEDDING_MODEL is not configured")
        if not settings.embedding_dim:
            raise EmbeddingConfigError("EMBEDDING_DIM is not configured")

        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key.get_secret_value()
        self.model = settings.embedding_model
        self.embedding_dim = settings.embedding_dim
        self._owned_client = client is None
        self.client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    def close(self) -> None:
        if self._owned_client:
            self.client.close()

    def embed_texts(self, texts: Sequence[str], *, task: str) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
            "dimensions": self.embedding_dim,
        }
        response = self._post_with_retries(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        response.raise_for_status()
        vectors = self._parse_openai_embedding_response(response.json(), expected_count=len(texts))
        return [self._prepare_vector(vector) for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], task="RETRIEVAL_QUERY")[0]

    def _post_with_retries(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        for attempt in range(MAX_EMBEDDING_RETRY_ATTEMPTS):
            response = self.client.post(url, headers=headers, json=json)
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt == MAX_EMBEDDING_RETRY_ATTEMPTS - 1:
                return response

            self._sleep(_retry_delay(response, attempt))

        return response

    def _parse_openai_embedding_response(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected_count:
            raise EmbeddingResponseError("Embedding response returned an unexpected number of vectors")

        ordered = sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
        vectors = []
        for item in ordered:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise EmbeddingResponseError("Embedding response item did not contain a vector")
            vectors.append(item["embedding"])
        return vectors

    def _prepare_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.embedding_dim:
            raise EmbeddingResponseError(
                f"Expected {self.embedding_dim} embedding dimensions, received {len(vector)}"
            )
        return normalize_vector([float(value) for value in vector])


def normalize_vector(vector: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        raise EmbeddingResponseError("Embedding vector had zero magnitude")
    return [float(value / length) for value in vector]


def build_missing_embeddings(
    session: Session,
    embedder: OpenAICompatibleEmbeddingClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> EmbeddingBatchResult:
    scanned = 0
    embedded = 0

    while limit is None or embedded < limit:
        remaining = None if limit is None else limit - embedded
        current_batch_size = batch_size if remaining is None else min(batch_size, remaining)
        if current_batch_size <= 0:
            break

        chunks = _chunks_needing_embeddings(
            session,
            model=embedder.model,
            dim=embedder.embedding_dim,
            limit=current_batch_size,
        )
        if not chunks:
            break

        scanned += len(chunks)
        vectors = embedder.embed_texts([chunk.text for chunk in chunks], task="RETRIEVAL_DOCUMENT")
        chunk_ids = [chunk.id for chunk in chunks]
        session.execute(delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids)))
        session.flush()

        for chunk, vector in zip(chunks, vectors, strict=True):
            session.add(
                Embedding(
                    chunk_id=chunk.id,
                    embedding_model=embedder.model,
                    embedding_dim=embedder.embedding_dim,
                    vector=vector,
                )
            )
        session.commit()
        embedded += len(chunks)

    return EmbeddingBatchResult(
        scanned_chunks=scanned,
        embedded_chunks=embedded,
        embedding_model=embedder.model,
        embedding_dim=embedder.embedding_dim,
    )


def retrieve_chunks(
    session: Session,
    embedder: OpenAICompatibleEmbeddingClient,
    query: str,
    *,
    filters: RetrievalFilters | None = None,
    limit: int = DEFAULT_RESULT_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    log_query: bool = True,
) -> RetrievalResult:
    clean_query = " ".join(query.split())
    active_filters = filters or RetrievalFilters()
    if not clean_query:
        return RetrievalResult(query=query, chunks=[], low_confidence=True)

    ranked = _rank_chunks(
        session,
        embedder=embedder,
        query=clean_query,
        filters=active_filters,
        limit=limit,
        candidate_limit=candidate_limit,
    )
    ranked = _with_neighboring_candidates(session, ranked, filters=active_filters, limit=limit)
    low_confidence = _is_low_confidence(ranked)

    if log_query:
        session.add(
            RetrievalLog(
                query_hash=_hash_query(clean_query),
                language=active_filters.language,
                filters=active_filters.as_log_dict(),
                retrieved_chunk_ids=[row.chunk.id for row in ranked],
                ranking=[
                    {
                        "chunk_id": str(row.chunk.id),
                        "score": row.score,
                        "vector_score": row.vector_score,
                        "keyword_score": row.keyword_score,
                    }
                    for row in ranked
                ],
                low_confidence=low_confidence,
            )
        )
        session.commit()

    if low_confidence:
        return RetrievalResult(query=clean_query, chunks=[], low_confidence=True)

    return RetrievalResult(
        query=clean_query,
        chunks=[row.to_retrieved_chunk() for row in ranked],
        low_confidence=False,
    )


def retrieve_chunks_with_language_fallback(
    session: Session,
    embedder: OpenAICompatibleEmbeddingClient,
    query: str,
    *,
    preferred_language: str | None,
    limit: int = DEFAULT_RESULT_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    log_query: bool = True,
) -> RetrievalResult:
    clean_query = " ".join(query.split())
    if not clean_query:
        return RetrievalResult(query=query, chunks=[], low_confidence=True)

    preferred_filters = RetrievalFilters(language=preferred_language) if preferred_language else RetrievalFilters()
    ranked = _rank_chunks(
        session,
        embedder=embedder,
        query=clean_query,
        filters=preferred_filters,
        limit=limit,
        candidate_limit=candidate_limit,
    )
    ranked = _with_neighboring_candidates(session, ranked, filters=preferred_filters, limit=limit)
    language_fallback_used = False

    if preferred_language and _is_low_confidence(ranked):
        fallback_ranked = _rank_chunks(
            session,
            embedder=embedder,
            query=clean_query,
            filters=RetrievalFilters(),
            limit=limit,
            candidate_limit=candidate_limit,
        )
        fallback_ranked = _with_neighboring_candidates(
            session,
            fallback_ranked,
            filters=RetrievalFilters(),
            limit=limit,
        )
        if not _is_low_confidence(fallback_ranked):
            ranked = fallback_ranked
            language_fallback_used = True

    low_confidence = _is_low_confidence(ranked)
    if log_query:
        filters_for_log = preferred_filters.as_log_dict()
        filters_for_log["language_fallback_used"] = language_fallback_used
        session.add(
            RetrievalLog(
                query_hash=_hash_query(clean_query),
                language=preferred_language,
                filters=filters_for_log,
                retrieved_chunk_ids=[row.chunk.id for row in ranked],
                ranking=[
                    {
                        "chunk_id": str(row.chunk.id),
                        "score": row.score,
                        "vector_score": row.vector_score,
                        "keyword_score": row.keyword_score,
                    }
                    for row in ranked
                ],
                low_confidence=low_confidence,
            )
        )
        session.commit()

    if low_confidence:
        return RetrievalResult(
            query=clean_query,
            chunks=[],
            low_confidence=True,
            language_fallback_used=language_fallback_used,
        )

    return RetrievalResult(
        query=clean_query,
        chunks=[row.to_retrieved_chunk() for row in ranked],
        low_confidence=False,
        language_fallback_used=language_fallback_used,
    )


def expand_retrieval_terms(query: str) -> str:
    normalized_query = " ".join(query.split())
    folded = normalized_query.casefold()
    additions: list[str] = []
    seen = {normalized_query.casefold()}
    for term, translations in RETRIEVAL_GLOSSARY.items():
        if term not in folded:
            continue
        for translation in translations:
            key = translation.casefold()
            if key not in seen:
                additions.append(translation)
                seen.add(key)
    if not additions:
        return normalized_query
    return " ".join([normalized_query, *additions])


@dataclass
class _Candidate:
    chunk: DocumentChunk
    source: Source
    vector_score: float = 0.0
    keyword_score: float = 0.0

    @property
    def score(self) -> float:
        both_bonus = 0.04 if self.vector_score > 0 and self.keyword_score > 0 else 0.0
        return (VECTOR_WEIGHT * self.vector_score) + (KEYWORD_WEIGHT * self.keyword_score) + both_bonus

    def to_retrieved_chunk(self) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(self.chunk.id),
            source_id=str(self.source.id),
            source_owner=self.source.owner,
            source_url=self.source.url,
            category=self.source.category,
            language=self.chunk.language,
            section_heading=self.chunk.section_heading,
            section_url=self.chunk.section_url,
            text=self.chunk.text,
            collected_at=self.chunk.collected_at,
            official_last_updated_at=self.chunk.official_last_updated_at,
            score=round(self.score, 6),
            vector_score=round(self.vector_score, 6),
            keyword_score=round(self.keyword_score, 6),
        )


def _rank_chunks(
    session: Session,
    *,
    embedder: OpenAICompatibleEmbeddingClient,
    query: str,
    filters: RetrievalFilters,
    limit: int,
    candidate_limit: int,
) -> list[_Candidate]:
    query_vector = embedder.embed_query(query)
    vector_rows = _vector_candidates(
        session,
        query_vector=query_vector,
        filters=filters,
        limit=candidate_limit,
    )
    keyword_rows = _keyword_candidates(
        session,
        query=query,
        filters=filters,
        limit=candidate_limit,
    )
    return _merge_rankings(vector_rows, keyword_rows, limit=limit)


def _with_neighboring_candidates(
    session: Session,
    rows: Sequence[_Candidate],
    *,
    filters: RetrievalFilters,
    limit: int,
) -> list[_Candidate]:
    if not rows or not hasattr(session, "execute"):
        return list(rows)

    merged: dict[str, _Candidate] = {str(row.chunk.id): row for row in rows}
    expanded: list[_Candidate] = []
    expanded_ids: set[str] = set()
    for row in rows[:2]:
        expanded.append(row)
        expanded_ids.add(str(row.chunk.id))
        for neighbor in _neighbor_candidates(session, row, filters=filters):
            key = str(neighbor.chunk.id)
            if key not in merged:
                merged[key] = neighbor
                expanded.append(neighbor)
                expanded_ids.add(key)

    for row in rows[2:]:
        if str(row.chunk.id) not in expanded_ids:
            expanded.append(row)
            expanded_ids.add(str(row.chunk.id))

    return sorted(expanded, key=lambda row: row.score, reverse=True)[:limit]


def _neighbor_candidates(session: Session, row: _Candidate, *, filters: RetrievalFilters) -> list[_Candidate]:
    statement = (
        select(DocumentChunk, Source)
        .join(Source, Source.id == DocumentChunk.source_id)
        .where(
            DocumentChunk.document_id == row.chunk.document_id,
            DocumentChunk.chunk_index.in_([row.chunk.chunk_index - 1, row.chunk.chunk_index + 1]),
        )
        .order_by(DocumentChunk.chunk_index)
    )
    statement = _apply_filters(statement, filters)

    return [
        _Candidate(
            chunk=chunk,
            source=source,
            vector_score=max(0.01, row.vector_score * 0.78),
            keyword_score=0.0,
        )
        for chunk, source in session.execute(statement).all()
    ]


def _chunks_needing_embeddings(
    session: Session,
    *,
    model: str,
    dim: int,
    limit: int,
) -> list[DocumentChunk]:
    return list(
        session.scalars(
            select(DocumentChunk)
            .outerjoin(Embedding)
            .where(
                or_(
                    Embedding.id.is_(None),
                    Embedding.embedding_model != model,
                    Embedding.embedding_dim != dim,
                )
            )
            .order_by(DocumentChunk.collected_at, DocumentChunk.chunk_index)
            .limit(limit)
        )
    )


def _vector_candidates(
    session: Session,
    *,
    query_vector: Sequence[float],
    filters: RetrievalFilters,
    limit: int,
) -> list[_Candidate]:
    distance = Embedding.vector.cosine_distance(query_vector).label("distance")
    statement = (
        select(DocumentChunk, Source, distance)
        .join(Embedding, Embedding.chunk_id == DocumentChunk.id)
        .join(Source, Source.id == DocumentChunk.source_id)
        .where(Embedding.embedding_dim == len(query_vector))
        .order_by(distance)
        .limit(limit)
    )
    statement = _apply_filters(statement, filters)

    rows = session.execute(statement).all()
    candidates: list[_Candidate] = []
    for chunk, source, raw_distance in rows:
        distance_value = float(raw_distance)
        similarity = max(0.0, 1.0 - distance_value)
        candidates.append(_Candidate(chunk=chunk, source=source, vector_score=similarity))
    return candidates


def _keyword_candidates(
    session: Session,
    *,
    query: str,
    filters: RetrievalFilters,
    limit: int,
) -> list[_Candidate]:
    tsquery = func.websearch_to_tsquery("simple", query)
    rank = func.ts_rank_cd(DocumentChunk.search_vector, tsquery).label("rank")
    statement = (
        select(DocumentChunk, Source, rank)
        .join(Source, Source.id == DocumentChunk.source_id)
        .where(DocumentChunk.search_vector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    statement = _apply_filters(statement, filters)

    rows = session.execute(statement).all()
    max_rank = max((float(raw_rank) for _, _, raw_rank in rows), default=0.0)
    if max_rank <= 0:
        return []

    return [
        _Candidate(
            chunk=chunk,
            source=source,
            keyword_score=min(1.0, float(raw_rank) / max_rank),
        )
        for chunk, source, raw_rank in rows
    ]


def _apply_filters(statement: Select[Any], filters: RetrievalFilters) -> Select[Any]:
    if filters.language:
        statement = statement.where(DocumentChunk.language == filters.language)
    if filters.owners:
        statement = statement.where(Source.owner.in_(filters.owners))
    if filters.categories:
        statement = statement.where(Source.category.in_(filters.categories))
    if filters.source_urls:
        statement = statement.where(Source.url.in_(filters.source_urls))
    return statement


def _merge_rankings(
    vector_rows: Iterable[_Candidate],
    keyword_rows: Iterable[_Candidate],
    *,
    limit: int,
) -> list[_Candidate]:
    merged: dict[str, _Candidate] = {}

    for row in vector_rows:
        merged[str(row.chunk.id)] = row

    for row in keyword_rows:
        key = str(row.chunk.id)
        existing = merged.get(key)
        if existing:
            existing.keyword_score = max(existing.keyword_score, row.keyword_score)
        else:
            merged[key] = row

    return sorted(merged.values(), key=lambda row: row.score, reverse=True)[:limit]


def _is_low_confidence(rows: Sequence[_Candidate]) -> bool:
    if not rows:
        return True
    best = rows[0]
    if best.keyword_score == 0:
        return best.vector_score < MIN_VECTOR_ONLY_SIMILARITY
    return best.score < MIN_RETRIEVAL_SCORE or (
        best.vector_score < MIN_VECTOR_SIMILARITY and best.keyword_score < 0.5
    )


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.casefold().encode("utf-8")).hexdigest()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(MAX_RETRY_SECONDS, max(0.0, float(retry_after)))
        except ValueError:
            pass

    return min(MAX_RETRY_SECONDS, INITIAL_RETRY_SECONDS * (2**attempt))


def build_embeddings_from_settings(
    settings: Settings,
    *,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingBatchResult:
    engine = create_engine_from_settings(settings)
    embedder = OpenAICompatibleEmbeddingClient(settings)
    try:
        with Session(engine) as session:
            return build_missing_embeddings(session, embedder, limit=limit, batch_size=batch_size)
    finally:
        embedder.close()
        engine.dispose()


def retrieve_from_settings(
    settings: Settings,
    query: str,
    *,
    filters: RetrievalFilters | None = None,
    limit: int = DEFAULT_RESULT_LIMIT,
) -> RetrievalResult:
    engine = create_engine_from_settings(settings)
    embedder = OpenAICompatibleEmbeddingClient(settings)
    try:
        with Session(engine) as session:
            return retrieve_chunks(session, embedder, query, filters=filters, limit=limit)
    finally:
        embedder.close()
        engine.dispose()


def embeddings_main() -> None:
    parser = argparse.ArgumentParser(description="Create or refresh embeddings for stored source chunks.")
    parser.add_argument("--limit", type=int, default=None, help="Only embed this many chunks.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Chunks to embed per commit.")
    args = parser.parse_args()

    result = build_embeddings_from_settings(
        get_settings(),
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(result.__dict__, indent=2))


def retrieval_main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval against the stored official source snapshot.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--language", choices=["en", "no"], default=None)
    parser.add_argument("--owner", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--source-url", action="append", default=[])
    parser.add_argument("--limit", type=int, default=DEFAULT_RESULT_LIMIT)
    args = parser.parse_args()

    filters = RetrievalFilters(
        language=args.language,
        owners=tuple(args.owner),
        categories=tuple(args.category),
        source_urls=tuple(args.source_url),
    )
    result = retrieve_from_settings(get_settings(), args.query, filters=filters, limit=args.limit)
    print(
        json.dumps(
            {
                "query": result.query,
                "low_confidence": result.low_confidence,
                "chunks": [
                    {
                        "source_owner": chunk.source_owner,
                        "source_url": chunk.source_url,
                        "section_heading": chunk.section_heading,
                        "section_url": chunk.section_url,
                        "language": chunk.language,
                        "score": chunk.score,
                        "vector_score": chunk.vector_score,
                        "keyword_score": chunk.keyword_score,
                        "collected_at": chunk.collected_at.isoformat(),
                        "official_last_updated_at": (
                            chunk.official_last_updated_at.isoformat()
                            if chunk.official_last_updated_at
                            else None
                        ),
                        "text_preview": chunk.text[:320],
                    }
                    for chunk in result.chunks
                ],
            },
            indent=2,
        )
    )
