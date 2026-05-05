from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from oslo_newcomer_rag.config import Settings
from oslo_newcomer_rag.evaluation import (
    CaseMetrics,
    EvalReport,
    EvalThresholds,
    GoldCase,
    JudgeScores,
    StaticJudge,
    answer_keyword_score,
    citation_coverage,
    context_precision,
    context_recall,
    evaluate_case,
    language_match,
    load_gold_dataset,
    parse_judge_response,
    refusal_correctness,
    run_live_evaluation,
    write_reports,
)
from oslo_newcomer_rag.generation import Citation, DataCurrency, GroundedAnswer
from oslo_newcomer_rag.retrieval import RetrievalResult, RetrievedChunk


def test_gold_dataset_loads_current_release_cases() -> None:
    dataset = load_gold_dataset(Path("eval/gold_questions.yml"))

    case_ids = {case.id for case in dataset.cases}

    assert dataset.version == 1
    assert "skilled_worker_basics" in case_ids
    assert "unsupported_second_hand_furniture" in case_ids
    assert "mixed_language_tax_card" in case_ids
    assert "permanent_residence_legal_risk" in case_ids
    assert dataset.thresholds.citation_coverage == 0.95


def test_context_metrics_compare_retrieved_chunks_with_expected_sources() -> None:
    case = _case(expected_owners=("UDI",), expected_categories=("immigration",))
    retrieval = RetrievalResult(
        query=case.question,
        low_confidence=False,
        chunks=[
            _chunk(chunk_id="chunk-1", owner="UDI", category="immigration"),
            _chunk(chunk_id="chunk-2", owner="NAV", category="welfare"),
        ],
    )

    assert context_precision(case, retrieval) == 0.5
    assert context_recall(case, retrieval) == 1.0


def test_answer_metrics_cover_citations_refusals_keywords_and_language() -> None:
    case = _case(expected_legal_disclaimer=True)
    retrieval = RetrievalResult(query=case.question, low_confidence=False, chunks=[_chunk()])
    answer = _answer(
        answer_text="UDI explains the appeal route for rejected applications. [S1]",
        citations=[_citation()],
        disclaimer="This is general information from official sources, not legal advice.",
    )

    assert citation_coverage(answer, retrieval) == 1.0
    assert refusal_correctness(case, answer) == 1.0
    assert answer_keyword_score(answer.answer, ("appeal", "rejected")) == 1.0
    assert language_match(answer.answer, "en") == 1.0
    assert language_match("Dette er generell informasjon fra offentlige kilder.", "no") == 1.0


def test_evaluate_case_blends_deterministic_and_judge_scores() -> None:
    case = _case(answer_keywords=("appeal", "UDI"))
    retrieval = RetrievalResult(query=case.question, low_confidence=False, chunks=[_chunk()])
    answer = _answer("UDI explains the appeal route. [S1]", citations=[_citation()])

    result = evaluate_case(
        case=case,
        retrieval=retrieval,
        answer=answer,
        judge=StaticJudge(JudgeScores(faithfulness=0.8, answer_relevance=0.6, notes="ok")),
    )

    assert result.metrics.faithfulness == 0.9
    assert result.metrics.answer_relevance == 0.8
    assert result.judge_scores.notes == "ok"


def test_parse_judge_response_accepts_json_blocks_and_clamps_scores() -> None:
    scores = parse_judge_response(
        '```json\n{"faithfulness": 1.3, "answer_relevance": -0.2, "notes": "bounded"}\n```'
    )

    assert scores.faithfulness == 1.0
    assert scores.answer_relevance == 0.0
    assert scores.notes == "bounded"


def test_report_writing_and_threshold_failure(tmp_path: Path) -> None:
    case = _case()
    result = evaluate_case(
        case=case,
        retrieval=RetrievalResult(query=case.question, low_confidence=False, chunks=[_chunk()]),
        answer=_answer("UDI explains the route. [S1]", citations=[_citation()]),
        judge=StaticJudge(JudgeScores(faithfulness=0.3, answer_relevance=0.3, notes="weak")),
    )
    report = EvalReport(
        generated_at=datetime(2026, 5, 5, 12, 0, tzinfo=UTC),
        dataset_path="eval/gold_questions.yml",
        thresholds=EvalThresholds(faithfulness=0.8, answer_relevance=0.8),
        results=(result,),
    )

    json_path, md_path = write_reports(report, tmp_path)

    assert report.passed is False
    assert json.loads(json_path.read_text(encoding="utf-8"))["passed"] is False
    assert "RAG Evaluation Report" in md_path.read_text(encoding="utf-8")


def test_live_evaluation_does_not_store_query_or_answer_text(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "gold.yml"
    dataset_path.write_text(
        """
version: 1
cases:
  - id: one
    question: What does UDI say about skilled workers?
    ui_language: en
    retrieval_language: en
    expected_refusal: false
    expected_owners: [UDI]
    expected_categories: [immigration]
    answer_keywords: [UDI]
""".strip(),
        encoding="utf-8",
    )
    retrieved = RetrievalResult(
        query="What does UDI say about skilled workers?",
        low_confidence=False,
        chunks=[_chunk()],
    )
    calls = {"log_query": None, "stored": False}

    class FakeEngine:
        def dispose(self) -> None:
            pass

    class FakeClient:
        def close(self) -> None:
            pass

    class FakeSession:
        def __init__(self, engine: FakeEngine) -> None:
            self.engine = engine

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def add(self, row) -> None:
            calls["stored"] = True

    def fake_retrieve(session, embedder, query, *, filters, log_query):
        calls["log_query"] = log_query
        return retrieved

    def fake_answer(*, question, ui_language, retrieval, chat_client, session_history=()):
        return _answer("UDI explains the skilled worker route. [S1]", citations=[_citation()])

    monkeypatch.setattr("oslo_newcomer_rag.evaluation.create_engine_from_settings", lambda settings: FakeEngine())
    monkeypatch.setattr("oslo_newcomer_rag.evaluation.Session", FakeSession)
    monkeypatch.setattr("oslo_newcomer_rag.evaluation.OpenAICompatibleEmbeddingClient", lambda settings: FakeClient())
    monkeypatch.setattr("oslo_newcomer_rag.evaluation.OpenAICompatibleChatClient", lambda settings: FakeClient())
    monkeypatch.setattr("oslo_newcomer_rag.evaluation.LlmJudge", lambda chat_client: StaticJudge(JudgeScores(1, 1, "ok")))
    monkeypatch.setattr("oslo_newcomer_rag.evaluation.retrieve_chunks", fake_retrieve)
    monkeypatch.setattr("oslo_newcomer_rag.evaluation.build_grounded_answer", fake_answer)

    report = run_live_evaluation(
        settings=Settings(
            database_url="postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
            llm_base_url="https://provider.example/v1",
            llm_api_key="test-key",
            llm_model="test-chat",
            embedding_model="test-embedding",
            embedding_dim=3,
        ),
        dataset_path=dataset_path,
    )

    assert calls["log_query"] is False
    assert calls["stored"] is False
    assert report.results[0].answer.answer == "UDI explains the skilled worker route. [S1]"


def _case(
    *,
    expected_refusal: bool = False,
    expected_legal_disclaimer: bool = False,
    expected_owners: tuple[str, ...] = ("UDI",),
    expected_categories: tuple[str, ...] = ("immigration",),
    answer_keywords: tuple[str, ...] = (),
) -> GoldCase:
    return GoldCase(
        id="case",
        question="What should I check?",
        ui_language="en",
        retrieval_language="en",
        expected_refusal=expected_refusal,
        expected_legal_disclaimer=expected_legal_disclaimer,
        expected_owners=expected_owners,
        expected_categories=expected_categories,
        expected_source_urls=(),
        answer_keywords=answer_keywords,
    )


def _chunk(
    *,
    chunk_id: str = "chunk-1",
    owner: str = "UDI",
    category: str = "immigration",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="source-1",
        source_owner=owner,
        source_url="https://www.udi.no/en/want-to-apply/",
        category=category,
        language="en",
        section_heading="Skilled workers",
        section_url="https://www.udi.no/en/want-to-apply/#skilled-workers",
        text="Skilled workers should check the official UDI route before they apply.",
        collected_at=datetime(2026, 2, 1, tzinfo=UTC),
        official_last_updated_at=datetime(2026, 1, 20, tzinfo=UTC),
        score=0.9,
        vector_score=0.8,
        keyword_score=1.0,
    )


def _citation() -> Citation:
    return Citation(
        citation_id="S1",
        chunk_id="chunk-1",
        source_owner="UDI",
        source_url="https://www.udi.no/en/want-to-apply/",
        section_url="https://www.udi.no/en/want-to-apply/#skilled-workers",
        section_heading="Skilled workers",
        collected_at=datetime(2026, 2, 1, tzinfo=UTC),
        official_last_updated_at=datetime(2026, 1, 20, tzinfo=UTC),
    )


def _answer(
    answer_text: str,
    *,
    citations: Sequence[Citation],
    disclaimer: str | None = None,
) -> GroundedAnswer:
    return GroundedAnswer(
        answer_id="answer-1",
        answer=f"{answer_text}\n\n{disclaimer}" if disclaimer else answer_text,
        refused=False,
        disclaimer=disclaimer,
        citations=list(citations),
        data_currency=DataCurrency(
            collected_at=datetime(2026, 2, 1, tzinfo=UTC),
            official_last_updated_at=datetime(2026, 1, 20, tzinfo=UTC),
        ),
    )
