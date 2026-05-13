from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx

from oslo_newcomer_rag.config import Settings
from oslo_newcomer_rag.db.models import DocumentChunk, Source
from oslo_newcomer_rag.retrieval import (
    OpenAICompatibleEmbeddingClient,
    RetrievalFilters,
    _Candidate,
    expand_retrieval_terms,
    _is_low_confidence,
    _merge_rankings,
    normalize_vector,
    retrieve_chunks_with_language_fallback,
)


def test_embedding_request_uses_configured_dimension_and_normalizes_vector() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [3.0, 4.0, 0.0],
                    }
                ]
            },
        )

    client = OpenAICompatibleEmbeddingClient(
        _settings(embedding_dim=3),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vector = client.embed_query("skilled worker permit")

    assert seen_payloads[0]["dimensions"] == 3
    assert seen_payloads[0]["model"] == "test-embedding"
    assert vector == [0.6, 0.8, 0.0]


def test_embedding_client_retries_rate_limited_requests() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0.25"}, json={"error": "slow down"})

        payload = json.loads(request.content)
        assert payload["dimensions"] == 3
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [0.0, 5.0, 0.0],
                    }
                ]
            },
        )

    client = OpenAICompatibleEmbeddingClient(
        _settings(embedding_dim=3),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
    )

    vector = client.embed_texts(["renew a residence card"])[0]

    assert calls == 2
    assert sleeps == [0.25]
    assert vector == [0.0, 1.0, 0.0]


def test_non_default_embedding_sizes_are_normalized_before_storage_or_scoring() -> None:
    vector = normalize_vector([10.0, 0.0, 0.0])

    assert vector == [1.0, 0.0, 0.0]


def test_hybrid_ranking_returns_expected_english_official_chunk() -> None:
    udi_chunk = _chunk(
        "en",
        "Skilled workers",
        "Skilled workers normally need a job offer and relevant qualifications.",
    )
    nav_chunk = _chunk(
        "en",
        "Social security membership",
        "NAV explains membership in the National Insurance Scheme.",
    )
    udi = _source("UDI", "https://www.udi.no/en/want-to-apply/work-immigration/", "permits")
    nav = _source("NAV", "https://www.nav.no/en/home", "welfare")

    ranked = _merge_rankings(
        [
            _Candidate(chunk=udi_chunk, source=udi, vector_score=0.81),
            _Candidate(chunk=nav_chunk, source=nav, vector_score=0.35),
        ],
        [_Candidate(chunk=udi_chunk, source=udi, keyword_score=1.0)],
        limit=2,
    )

    assert ranked[0].source.owner == "UDI"
    assert ranked[0].chunk.section_heading == "Skilled workers"
    assert _is_low_confidence(ranked) is False


def test_hybrid_ranking_handles_norwegian_wording() -> None:
    tax_chunk = _chunk(
        "no",
        "Skattekort",
        "Du trenger skattekort når du skal jobbe i Norge.",
    )
    sio_chunk = _chunk(
        "no",
        "Studentbolig",
        "SiO har informasjon om bolig for studenter i Oslo.",
    )
    tax = _source("Skatteetaten", "https://www.skatteetaten.no/person/", "tax")
    sio = _source("SiO", "https://www.sio.no/", "student")

    ranked = _merge_rankings(
        [
            _Candidate(chunk=tax_chunk, source=tax, vector_score=0.74),
            _Candidate(chunk=sio_chunk, source=sio, vector_score=0.22),
        ],
        [_Candidate(chunk=tax_chunk, source=tax, keyword_score=1.0)],
        limit=2,
    )

    result = ranked[0].to_retrieved_chunk()

    assert result.source_owner == "Skatteetaten"
    assert result.language == "no"
    assert result.section_heading == "Skattekort"


def test_retrieval_glossary_adds_english_terms_for_norwegian_queries() -> None:
    expanded = expand_retrieval_terms("Hvordan får jeg skattekort og D-nummer?")

    assert "tax deduction card" in expanded
    assert "D number" in expanded
    assert "identification number" in expanded


def test_language_fallback_returns_english_chunks_when_norwegian_filter_is_weak(monkeypatch) -> None:
    tax_chunk = _chunk(
        "en",
        "Tax deduction card",
        "You need a tax deduction card when you work in Norway.",
    )
    tax_source = _source("Skatteetaten", "https://www.skatteetaten.no/en/person/", "tax")
    calls: list[str | None] = []

    def fake_rank(session, *, embedder, query, filters, limit, candidate_limit):
        calls.append(filters.language)
        if filters.language == "no":
            return []
        return [_Candidate(chunk=tax_chunk, source=tax_source, vector_score=0.78, keyword_score=1.0)]

    monkeypatch.setattr("oslo_newcomer_rag.retrieval._rank_chunks", fake_rank)

    result = retrieve_chunks_with_language_fallback(
        session=object(),
        embedder=object(),
        query="Hvordan får jeg skattekort?",
        preferred_language="no",
        log_query=False,
    )

    assert calls == ["no", None]
    assert result.low_confidence is False
    assert result.language_fallback_used is True
    assert result.chunks[0].source_owner == "Skatteetaten"


def test_low_confidence_result_is_refused_for_unrelated_query() -> None:
    chunk = _chunk("en", "Parking permits", "Residential parking information for Oslo.")
    source = _source("Oslo kommune", "https://www.oslo.kommune.no/english/", "municipal")
    ranked = [_Candidate(chunk=chunk, source=source, vector_score=0.08, keyword_score=0.0)]

    assert _is_low_confidence(ranked) is True


def test_source_filters_are_serialized_for_retrieval_logs() -> None:
    filters = RetrievalFilters(
        language="en",
        owners=("UDI",),
        categories=("permits",),
        source_urls=("https://www.udi.no/en/",),
    )

    assert filters.as_log_dict() == {
        "language": "en",
        "owners": ["UDI"],
        "categories": ["permits"],
        "source_urls": ["https://www.udi.no/en/"],
    }


def _settings(
    *,
    llm_base_url: str = "https://provider.example/v1",
    embedding_model: str = "test-embedding",
    embedding_dim: int = 3,
) -> Settings:
    return Settings(
        llm_base_url=llm_base_url,
        llm_api_key="test-key",
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )


def _source(owner: str, url: str, category: str) -> Source:
    return Source(
        id=uuid.uuid4(),
        owner=owner,
        url=url,
        language="en",
        category=category,
        intended_coverage={},
    )


def _chunk(language: str, heading: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        chunk_index=0,
        section_heading=heading,
        section_url="https://example.test/section",
        language=language,
        text=text,
        text_hash="hash",
        token_count=len(text.split()),
        collected_at=datetime(2026, 2, 1, tzinfo=UTC),
        official_last_updated_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
