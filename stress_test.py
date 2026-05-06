from __future__ import annotations

import re
import sys
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Literal

from fastapi.testclient import TestClient

from oslo_newcomer_rag.main import create_app


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatStep:
    session_id: str
    question: str
    ui_language: Literal["en", "no"]
    label: str
    must_refuse: bool = False
    needs_disclaimer: bool = False
    expected_language: Literal["en", "no"] | None = None
    expected_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    needs_citations: bool = True
    max_avg_sentence_words: int | None = None


@dataclass
class StepResult:
    step: ChatStep
    response: dict
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def main() -> int:
    client = TestClient(create_app())
    histories: dict[str, list[ChatMessage]] = {}
    results: list[StepResult] = []

    for step in build_suite():
        history = histories.setdefault(step.session_id, [])
        payload = {
            "question": step.question,
            "ui_language": step.ui_language,
            "session_history": [message.__dict__ for message in history],
        }
        response = client.post("/api/chat", json=payload)
        if response.status_code != 200:
            print(f"\nFAIL {step.label}: HTTP {response.status_code}")
            print(response.text)
            return 1

        data = response.json()
        result = evaluate_step(step, data)
        results.append(result)
        history.extend(
            [
                ChatMessage(role="user", content=step.question),
                ChatMessage(role="assistant", content=data["answer"]),
            ]
        )
        print_result(result)

    passed = sum(1 for result in results if result.passed)
    print(f"\nSummary: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


def build_suite() -> list[ChatStep]:
    memory_session = f"memory-{uuid.uuid4()}"
    study_session = f"study-{uuid.uuid4()}"
    refusal_session = f"refusal-{uuid.uuid4()}"
    edge_session = f"edge-{uuid.uuid4()}"

    return [
        ChatStep(
            session_id=memory_session,
            label="memory: D-number",
            question="I just got a job in Oslo. How do I get a D-number?",
            ui_language="en",
            expected_language="en",
            expected_terms=("d number", "identification"),
        ),
        ChatStep(
            session_id=memory_session,
            label="memory: appointment follow-up",
            question="Do I have to book an appointment to get it?",
            ui_language="en",
            expected_language="en",
            expected_terms=("appointment", "residence permit"),
            forbidden_terms=("what you mean by",),
        ),
        ChatStep(
            session_id=memory_session,
            label="memory: SUA follow-up",
            question="Can I go to the SUA office for that?",
            ui_language="en",
            expected_language="en",
            expected_terms=("sua", "appointment"),
            forbidden_terms=("what you mean by",),
        ),
        ChatStep(
            session_id=study_session,
            label="norwegian: study permit",
            question="Hvordan søker jeg om studietillatelse i Norge?",
            ui_language="no",
            expected_language="no",
            expected_terms=("studietillatelse", "udi"),
            max_avg_sentence_words=24,
        ),
        ChatStep(
            session_id=study_session,
            label="norwegian: funds follow-up",
            question="Hvor mye penger må jeg bevise at jeg har?",
            ui_language="no",
            expected_language="no",
            expected_terms=("penger", "udi"),
            forbidden_terms=("jobbsøker", "arbeidsinnvandring", "phd", "325 400", "27 116"),
            max_avg_sentence_words=24,
        ),
        ChatStep(
            session_id=refusal_session,
            label="refusal: appeal letter",
            question=(
                "I received a letter from UDI saying my work permit was rejected. "
                "Can you write an appeal letter for me to send to them?"
            ),
            ui_language="en",
            must_refuse=True,
            needs_disclaimer=True,
            expected_language="en",
            expected_terms=("cannot", "appeal"),
            needs_citations=False,
        ),
        ChatStep(
            session_id=refusal_session,
            label="refusal: cheap bars",
            question="What are the best cheap bars in Grünerløkka?",
            ui_language="en",
            must_refuse=True,
            expected_language="en",
            expected_terms=("official",),
            needs_citations=False,
        ),
        ChatStep(
            session_id=edge_session,
            label="edge: EU dagpenger",
            question=(
                "I am an EU citizen looking for work in Oslo. "
                "Am I eligible for unemployment benefits (dagpenger) right now?"
            ),
            ui_language="en",
            needs_disclaimer=True,
            expected_language="en",
            expected_terms=("nav", "jobseeker", "dagpenger"),
            forbidden_terms=("you are eligible", "you qualify"),
        ),
    ]


def evaluate_step(step: ChatStep, response: dict) -> StepResult:
    answer = str(response.get("answer") or "")
    folded = answer.casefold()
    result = StepResult(step=step, response=response)

    result.checks.append(
        ("refusal state", bool(response.get("refused")) is step.must_refuse, f"expected {step.must_refuse}")
    )
    if step.expected_language:
        result.checks.append(
            (
                "answer language",
                answer_language(answer) == step.expected_language,
                f"expected {step.expected_language}",
            )
        )
    if step.needs_disclaimer:
        disclaimer = str(response.get("disclaimer") or "")
        result.checks.append(("disclaimer", "legal advice" in disclaimer.casefold(), "missing legal disclaimer"))
        result.checks.append(("no duplicate disclaimer", disclaimer not in answer, "disclaimer repeated in answer"))
    if step.expected_terms:
        missing = [term for term in step.expected_terms if term not in folded]
        result.checks.append(("expected terms", not missing, f"missing: {', '.join(missing)}"))
    if step.forbidden_terms:
        found = [term for term in step.forbidden_terms if term in folded]
        result.checks.append(("forbidden terms", not found, f"found: {', '.join(found)}"))
    if step.needs_citations and not bool(response.get("refused")):
        result.checks.append(("citations present", bool(response.get("citations")), "no citation objects"))
        result.checks.append(("claim citations", claims_are_cited(answer), "uncited factual line"))
    if step.max_avg_sentence_words:
        result.checks.append(
            (
                "simple wording",
                avg_sentence_words(answer) <= step.max_avg_sentence_words,
                f"average sentence length: {avg_sentence_words(answer):.1f}",
            )
        )
    return result


def print_result(result: StepResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    step = result.step
    response = result.response
    print(f"\n[{status}] {step.label} ({step.session_id})")
    print(f"Q: {step.question}")
    print(f"refused={response.get('refused')} citations={len(response.get('citations') or [])}")
    if response.get("disclaimer"):
        print(f"disclaimer: {response['disclaimer']}")
    print("answer:")
    print(textwrap.indent(textwrap.fill(response.get("answer", ""), width=96), "  "))
    for name, ok, detail in result.checks:
        marker = "ok" if ok else "fail"
        print(f"  - {marker}: {name} ({detail})")


def claims_are_cited(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    factual_lines = [
        line
        for line in lines
        if len(line.split()) >= 5
        and not line.startswith(("Note", "Merk"))
        and "not legal advice" not in line.casefold()
        and "juridisk rådgivning" not in line.casefold()
    ]
    return bool(factual_lines) and all(re.search(r"\[S\d+\]", line) for line in factual_lines)


def answer_language(answer: str) -> str:
    folded = f" {answer.casefold()} "
    no_markers = (" jeg ", " du ", " det ", " ikke ", " offentlige ", " søke ", " penger ", " tillatelse ")
    en_markers = (" you ", " the ", " can ", " check ", " official ", " not enough ", " appointment ")
    no_score = sum(marker in folded for marker in no_markers)
    en_score = sum(marker in folded for marker in en_markers)
    return "no" if no_score > en_score else "en"


def avg_sentence_words(answer: str) -> float:
    clean = re.sub(r"\[S\d+\]", "", answer)
    sentences = [part.strip() for part in re.split(r"[.!?]\s+", clean) if part.strip()]
    if not sentences:
        return 0.0
    return sum(len(sentence.split()) for sentence in sentences) / len(sentences)


if __name__ == "__main__":
    sys.exit(main())
