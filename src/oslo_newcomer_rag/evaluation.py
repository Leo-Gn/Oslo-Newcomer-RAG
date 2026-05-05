from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml
from sqlalchemy.orm import Session

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.db.session import create_engine_from_settings
from oslo_newcomer_rag.generation import (
    ChatClient,
    ChatMessage,
    Citation,
    GroundedAnswer,
    OpenAICompatibleChatClient,
    build_grounded_answer,
)
from oslo_newcomer_rag.retrieval import (
    OpenAICompatibleEmbeddingClient,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
    retrieve_chunks,
)


DEFAULT_DATASET_PATH = Path("eval/gold_questions.yml")
DEFAULT_REPORTS_DIR = Path("eval_reports")
JUDGE_CONTEXT_CHARS = 7000
CITATION_PATTERN = re.compile(r"\[S\d+\]")


@dataclass(frozen=True)
class EvalThresholds:
    context_precision: float = 0.60
    context_recall: float = 0.60
    faithfulness: float = 0.80
    answer_relevance: float = 0.75
    citation_coverage: float = 0.95
    refusal_correctness: float = 1.00
    language_match: float = 0.90

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "EvalThresholds":
        if not values:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown threshold field(s): {joined}")
        return cls(**{key: float(value) for key, value in values.items()})

    def as_dict(self) -> dict[str, float]:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "citation_coverage": self.citation_coverage,
            "refusal_correctness": self.refusal_correctness,
            "language_match": self.language_match,
        }


@dataclass(frozen=True)
class GoldCase:
    id: str
    question: str
    ui_language: str
    retrieval_language: str
    expected_refusal: bool
    expected_legal_disclaimer: bool
    expected_owners: tuple[str, ...]
    expected_categories: tuple[str, ...]
    expected_source_urls: tuple[str, ...]
    answer_keywords: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "GoldCase":
        required = {"id", "question", "ui_language", "expected_refusal"}
        missing = required - set(values)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Gold case is missing required field(s): {joined}")

        ui_language = str(values["ui_language"]).lower()
        if ui_language not in {"en", "no"}:
            raise ValueError(f"{values['id']} has unsupported ui_language: {ui_language}")

        retrieval_language = str(values.get("retrieval_language") or ui_language).lower()
        if retrieval_language not in {"en", "no"}:
            raise ValueError(f"{values['id']} has unsupported retrieval_language: {retrieval_language}")

        case = cls(
            id=str(values["id"]),
            question=" ".join(str(values["question"]).split()),
            ui_language=ui_language,
            retrieval_language=retrieval_language,
            expected_refusal=bool(values["expected_refusal"]),
            expected_legal_disclaimer=bool(values.get("expected_legal_disclaimer", False)),
            expected_owners=_string_tuple(values.get("expected_owners", ())),
            expected_categories=_string_tuple(values.get("expected_categories", ())),
            expected_source_urls=_string_tuple(values.get("expected_source_urls", ())),
            answer_keywords=_string_tuple(values.get("answer_keywords", ())),
        )
        case.validate()
        return case

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("Gold case id cannot be empty")
        if not self.question.strip():
            raise ValueError(f"{self.id} question cannot be empty")
        if not self.expected_refusal and not self.expected_evidence:
            raise ValueError(f"{self.id} must name expected evidence for a supported answer")

    @property
    def expected_evidence(self) -> tuple[tuple[str, str], ...]:
        values: list[tuple[str, str]] = []
        values.extend(("owner", owner.casefold()) for owner in self.expected_owners)
        values.extend(("category", category.casefold()) for category in self.expected_categories)
        values.extend(("url", url.casefold()) for url in self.expected_source_urls)
        return tuple(values)


@dataclass(frozen=True)
class GoldDataset:
    version: int
    description: str
    thresholds: EvalThresholds
    cases: tuple[GoldCase, ...]


@dataclass(frozen=True)
class JudgeScores:
    faithfulness: float
    answer_relevance: float
    notes: str


@dataclass(frozen=True)
class CaseMetrics:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    citation_coverage: float
    refusal_correctness: float
    language_match: float

    def as_dict(self) -> dict[str, float]:
        return {
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "refusal_correctness": round(self.refusal_correctness, 4),
            "language_match": round(self.language_match, 4),
        }


@dataclass(frozen=True)
class CaseResult:
    case: GoldCase
    retrieval: RetrievalResult
    answer: GroundedAnswer
    metrics: CaseMetrics
    judge_scores: JudgeScores


@dataclass(frozen=True)
class EvalReport:
    generated_at: datetime
    dataset_path: str
    thresholds: EvalThresholds
    results: tuple[CaseResult, ...]

    @property
    def summary(self) -> CaseMetrics:
        if not self.results:
            return CaseMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        count = len(self.results)
        totals = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "citation_coverage": 0.0,
            "refusal_correctness": 0.0,
            "language_match": 0.0,
        }
        for result in self.results:
            for key, value in result.metrics.as_dict().items():
                totals[key] += value

        return CaseMetrics(**{key: value / count for key, value in totals.items()})

    @property
    def passed(self) -> bool:
        summary = self.summary
        thresholds = self.thresholds
        threshold_passed = all(
            (
                summary.context_precision >= thresholds.context_precision,
                summary.context_recall >= thresholds.context_recall,
                summary.faithfulness >= thresholds.faithfulness,
                summary.answer_relevance >= thresholds.answer_relevance,
                summary.citation_coverage >= thresholds.citation_coverage,
                summary.refusal_correctness >= thresholds.refusal_correctness,
                summary.language_match >= thresholds.language_match,
            )
        )
        refusal_cases_passed = all(
            result.metrics.refusal_correctness == 1.0
            for result in self.results
            if result.case.expected_refusal
        )
        return threshold_passed and refusal_cases_passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "dataset_path": self.dataset_path,
            "passed": self.passed,
            "thresholds": self.thresholds.as_dict(),
            "summary": self.summary.as_dict(),
            "cases": [_case_result_dict(result) for result in self.results],
        }


class JudgeClient(Protocol):
    def judge(self, *, question: str, answer: str, context: Sequence[RetrievedChunk]) -> JudgeScores:
        pass


class LlmJudge:
    def __init__(self, chat_client: ChatClient) -> None:
        self.chat_client = chat_client

    def judge(self, *, question: str, answer: str, context: Sequence[RetrievedChunk]) -> JudgeScores:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You grade a RAG answer for a small public-service demo. "
                    "Use only the supplied context. Return JSON only."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Score both fields from 0 to 1.\n"
                    "faithfulness: factual claims are supported by the context.\n"
                    "answer_relevance: the answer directly addresses the question.\n\n"
                    f"Question:\n{question}\n\n"
                    f"Answer:\n{answer}\n\n"
                    f"Context:\n{_judge_context(context)}\n\n"
                    'Return: {"faithfulness": 0.0, "answer_relevance": 0.0, "notes": "short reason"}'
                ),
            ),
        ]
        return parse_judge_response(self.chat_client.complete(messages))


class StaticJudge:
    def __init__(self, scores: JudgeScores) -> None:
        self.scores = scores

    def judge(self, *, question: str, answer: str, context: Sequence[RetrievedChunk]) -> JudgeScores:
        return self.scores


def load_gold_dataset(path: Path = DEFAULT_DATASET_PATH) -> GoldDataset:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Gold dataset must be a mapping")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("Gold dataset must contain at least one case")

    cases = tuple(GoldCase.from_mapping(case) for case in cases_raw)
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Gold case ids must be unique")

    return GoldDataset(
        version=int(raw.get("version", 1)),
        description=str(raw.get("description", "")),
        thresholds=EvalThresholds.from_mapping(raw.get("thresholds")),
        cases=cases,
    )


def parse_judge_response(content: str) -> JudgeScores:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Judge response was not valid JSON") from None
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")

    return JudgeScores(
        faithfulness=_clamp_score(payload.get("faithfulness")),
        answer_relevance=_clamp_score(payload.get("answer_relevance")),
        notes=str(payload.get("notes") or "").strip(),
    )


def evaluate_case(
    *,
    case: GoldCase,
    retrieval: RetrievalResult,
    answer: GroundedAnswer,
    judge: JudgeClient,
) -> CaseResult:
    judge_scores = _judge_or_default(case=case, retrieval=retrieval, answer=answer, judge=judge)
    deterministic_faithfulness = citation_coverage(answer, retrieval)
    keyword_relevance = answer_keyword_score(answer.answer, case.answer_keywords)

    metrics = CaseMetrics(
        context_precision=context_precision(case, retrieval),
        context_recall=context_recall(case, retrieval),
        faithfulness=_average(deterministic_faithfulness, judge_scores.faithfulness),
        answer_relevance=_average(keyword_relevance, judge_scores.answer_relevance),
        citation_coverage=citation_coverage(answer, retrieval),
        refusal_correctness=refusal_correctness(case, answer),
        language_match=language_match(answer.answer, case.ui_language),
    )
    return CaseResult(
        case=case,
        retrieval=retrieval,
        answer=answer,
        metrics=metrics,
        judge_scores=judge_scores,
    )


def run_live_evaluation(
    *,
    settings: Settings,
    dataset_path: Path,
    limit: int | None = None,
) -> EvalReport:
    dataset = load_gold_dataset(dataset_path)
    cases = dataset.cases[:limit] if limit else dataset.cases
    if not settings.has_database_config:
        raise RuntimeError("DATABASE_URL is not configured")

    engine = create_engine_from_settings(settings)
    embedder = OpenAICompatibleEmbeddingClient(settings)
    chat_client = OpenAICompatibleChatClient(settings)
    judge = LlmJudge(chat_client)
    results: list[CaseResult] = []

    try:
        with Session(engine) as session:
            for case in cases:
                retrieval = retrieve_chunks(
                    session,
                    embedder,
                    case.question,
                    filters=RetrievalFilters(language=case.retrieval_language),
                    log_query=False,
                )
                answer = build_grounded_answer(
                    question=case.question,
                    ui_language=case.ui_language,
                    retrieval=retrieval,
                    chat_client=chat_client,
                )
                results.append(
                    evaluate_case(
                        case=case,
                        retrieval=retrieval,
                        answer=answer,
                        judge=judge,
                    )
                )
    finally:
        embedder.close()
        chat_client.close()
        engine.dispose()

    return EvalReport(
        generated_at=datetime.now(UTC),
        dataset_path=str(dataset_path),
        thresholds=dataset.thresholds,
        results=tuple(results),
    )


def context_precision(case: GoldCase, retrieval: RetrievalResult) -> float:
    if retrieval.low_confidence or not retrieval.chunks:
        return 1.0 if case.expected_refusal else 0.0
    if not case.expected_evidence:
        return 0.0
    matches = sum(1 for chunk in retrieval.chunks if _chunk_matches_case(chunk, case))
    return matches / len(retrieval.chunks)


def context_recall(case: GoldCase, retrieval: RetrievalResult) -> float:
    if case.expected_refusal:
        return 1.0 if retrieval.low_confidence or not retrieval.chunks else 0.0
    expected = case.expected_evidence
    if not expected:
        return 0.0
    found = 0
    for evidence_type, value in expected:
        if any(_chunk_has_evidence(chunk, evidence_type, value) for chunk in retrieval.chunks):
            found += 1
    return found / len(expected)


def citation_coverage(answer: GroundedAnswer, retrieval: RetrievalResult) -> float:
    if answer.refused:
        return 1.0 if not answer.citations else 0.0
    if not answer.citations:
        return 0.0

    retrieved_ids = {chunk.chunk_id for chunk in retrieval.chunks}
    valid_citations = sum(1 for citation in answer.citations if citation.chunk_id in retrieved_ids)
    citation_score = valid_citations / len(answer.citations)

    factual_lines = _factual_answer_lines(answer.answer, answer.disclaimer)
    if not factual_lines:
        sentence_score = 0.0
    else:
        cited_lines = sum(1 for line in factual_lines if CITATION_PATTERN.search(line))
        sentence_score = cited_lines / len(factual_lines)

    return _average(citation_score, sentence_score)


def refusal_correctness(case: GoldCase, answer: GroundedAnswer) -> float:
    if answer.refused != case.expected_refusal:
        return 0.0
    if case.expected_legal_disclaimer and not answer.disclaimer:
        return 0.0
    return 1.0


def language_match(answer: str, ui_language: str) -> float:
    folded = answer.casefold()
    if ui_language == "no":
        norwegian_markers = (
            "jeg",
            "du",
            "det",
            "dette",
            "ikke",
            "offentlige",
            "kilder",
            "juridisk",
            "rådgivning",
            "skattekort",
            "opphold",
        )
        return 1.0 if any(marker in folded for marker in norwegian_markers) else 0.0

    english_markers = ("the", "you", "official", "source", "norway", "check", "not enough")
    norwegian_markers = (" ikke ", " juridisk ", " offentlige ", " skattekort ")
    if any(marker in f" {folded} " for marker in norwegian_markers):
        return 0.0
    return 1.0 if any(marker in folded for marker in english_markers) else 0.0


def answer_keyword_score(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    folded = answer.casefold()
    hits = sum(1 for keyword in keywords if keyword.casefold() in folded)
    return hits / len(keywords)


def write_reports(report: EvalReport, reports_dir: Path = DEFAULT_REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"rag-eval-{stamp}.json"
    md_path = reports_dir / f"rag-eval-{stamp}.md"

    json_path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    return json_path, md_path


def markdown_report(report: EvalReport) -> str:
    status = "passed" if report.passed else "failed"
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        f"Status: {status}",
        f"Dataset: {report.dataset_path}",
        "",
        "## Summary",
        "",
        "| Metric | Score | Threshold |",
        "| --- | ---: | ---: |",
    ]
    summary = report.summary.as_dict()
    thresholds = report.thresholds.as_dict()
    for metric, score in summary.items():
        lines.append(f"| {metric} | {score:.4f} | {thresholds[metric]:.4f} |")

    lines.extend(["", "## Cases", ""])
    for result in report.results:
        case_status = "passed" if _case_passed(result, report.thresholds) else "failed"
        lines.extend(
            [
                f"### {result.case.id}",
                "",
                f"- Status: {case_status}",
                f"- Refused: {str(result.answer.refused).lower()}",
                f"- Retrieved chunks: {len(result.retrieval.chunks)}",
                f"- Judge notes: {result.judge_scores.notes or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the stored Oslo newcomer RAG snapshot.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases.")
    args = parser.parse_args()

    report = run_live_evaluation(settings=get_settings(), dataset_path=args.dataset, limit=args.limit)
    json_path, md_path = write_reports(report, args.reports_dir)

    summary = report.summary.as_dict()
    print(f"RAG eval {'passed' if report.passed else 'failed'}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(json.dumps({"summary": summary, "passed": report.passed}, indent=2))

    if not report.passed:
        raise SystemExit(1)


def _judge_or_default(
    *,
    case: GoldCase,
    retrieval: RetrievalResult,
    answer: GroundedAnswer,
    judge: JudgeClient,
) -> JudgeScores:
    if answer.refused:
        score = 1.0 if case.expected_refusal else 0.0
        return JudgeScores(faithfulness=score, answer_relevance=score, notes="refusal checked deterministically")
    if retrieval.low_confidence or not retrieval.chunks:
        return JudgeScores(faithfulness=0.0, answer_relevance=0.0, notes="no context for judge")
    return judge.judge(question=case.question, answer=answer.answer, context=retrieval.chunks)


def _judge_context(chunks: Sequence[RetrievedChunk]) -> str:
    parts = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.text.split())
        block = (
            f"[S{index}] {chunk.source_owner} | {chunk.section_heading} | "
            f"{chunk.section_url}\n{text}\n"
        )
        if used_chars + len(block) > JUDGE_CONTEXT_CHARS:
            break
        parts.append(block)
        used_chars += len(block)
    return "\n".join(parts)


def _case_result_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "id": result.case.id,
        "question": result.case.question,
        "ui_language": result.case.ui_language,
        "retrieval_language": result.case.retrieval_language,
        "expected_refusal": result.case.expected_refusal,
        "refused": result.answer.refused,
        "metrics": result.metrics.as_dict(),
        "judge": {
            "faithfulness": round(result.judge_scores.faithfulness, 4),
            "answer_relevance": round(result.judge_scores.answer_relevance, 4),
            "notes": result.judge_scores.notes,
        },
        "answer": result.answer.answer,
        "citations": [_citation_dict(citation) for citation in result.answer.citations],
        "retrieved_chunks": [_chunk_dict(chunk) for chunk in result.retrieval.chunks],
    }


def _citation_dict(citation: Citation) -> dict[str, Any]:
    return {
        "citation_id": citation.citation_id,
        "chunk_id": citation.chunk_id,
        "source_owner": citation.source_owner,
        "source_url": citation.source_url,
        "section_url": citation.section_url,
        "section_heading": citation.section_heading,
        "collected_at": citation.collected_at.isoformat(),
        "official_last_updated_at": (
            citation.official_last_updated_at.isoformat()
            if citation.official_last_updated_at
            else None
        ),
    }


def _chunk_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_owner": chunk.source_owner,
        "source_url": chunk.source_url,
        "category": chunk.category,
        "language": chunk.language,
        "section_heading": chunk.section_heading,
        "section_url": chunk.section_url,
        "score": chunk.score,
        "vector_score": chunk.vector_score,
        "keyword_score": chunk.keyword_score,
    }


def _factual_answer_lines(answer: str, disclaimer: str | None) -> list[str]:
    if disclaimer:
        answer = answer.replace(disclaimer, "")
    rough_lines = re.split(r"\n+", answer)
    return [line.strip() for line in rough_lines if len(line.split()) >= 4]


def _chunk_matches_case(chunk: RetrievedChunk, case: GoldCase) -> bool:
    if case.expected_owners and chunk.source_owner in case.expected_owners:
        return True
    if case.expected_categories and chunk.category in case.expected_categories:
        return True
    if case.expected_source_urls and chunk.source_url in case.expected_source_urls:
        return True
    return False


def _chunk_has_evidence(chunk: RetrievedChunk, evidence_type: str, value: str) -> bool:
    if evidence_type == "owner":
        return chunk.source_owner.casefold() == value
    if evidence_type == "category":
        return chunk.category.casefold() == value
    if evidence_type == "url":
        return chunk.source_url.casefold() == value or chunk.section_url.casefold().startswith(value)
    return False


def _case_passed(result: CaseResult, thresholds: EvalThresholds) -> bool:
    metrics = result.metrics
    return all(
        (
            metrics.context_precision >= thresholds.context_precision,
            metrics.context_recall >= thresholds.context_recall,
            metrics.faithfulness >= thresholds.faithfulness,
            metrics.answer_relevance >= thresholds.answer_relevance,
            metrics.citation_coverage >= thresholds.citation_coverage,
            metrics.refusal_correctness >= thresholds.refusal_correctness,
            metrics.language_match >= thresholds.language_match,
        )
    )


def _average(left: float, right: float) -> float:
    return (left + right) / 2


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("Judge score must be numeric") from None
    return max(0.0, min(1.0, numeric))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("Expected a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())
