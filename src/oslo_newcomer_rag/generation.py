from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import httpx

from oslo_newcomer_rag.config import Settings
from oslo_newcomer_rag.retrieval import RetrievedChunk, RetrievalResult


MAX_CONTEXT_CHARS = 11_000
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
GROUPED_CITATION_PATTERN = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)+)\]")
ID_NUMBER_PATTERN = re.compile(r"\b\d{6}\s?\d{5}\b")
LEGAL_RISK_TERMS = (
    "appeal",
    "rejected",
    "rejection",
    "refused",
    "denied",
    "deport",
    "asylum",
    "eligible",
    "eligibility",
    "my case",
    "my application",
    "should i apply",
    "can i get",
    "klage",
    "avslag",
    "utvist",
    "asyl",
    "har jeg rett",
    "min sak",
    "søknaden min",
    "kan jeg få",
)
GREETING_TERMS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "hei",
    "heisann",
    "hallo",
    "god morgen",
    "god dag",
    "god kveld",
}
PERSONAL_RECORD_TERMS = (
    "tax records",
    "personal record",
    "personal records",
    "my record",
    "my records",
    "owe money",
    "debt",
    "case status",
    "application status",
    "check my application",
    "my personal id",
    "personal id number",
    "fødselsnummeret mitt",
    "personnummeret mitt",
    "skatteregister",
    "skattemelding",
    "sjekke skatten",
    "skylder penger",
    "gjeld",
    "status på saken",
    "status på søknaden",
)
LEGAL_DRAFTING_TERMS = (
    "write an appeal",
    "appeal letter",
    "write a letter",
    "draft an appeal",
    "fill the form",
    "fill out the form",
    "complete the form",
    "send to them",
    "klagebrev",
    "skrive klage",
    "fylle ut skjema",
    "fylle skjema",
)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    source_owner: str
    source_url: str
    section_url: str
    section_heading: str
    collected_at: datetime
    official_last_updated_at: datetime | None


@dataclass(frozen=True)
class DataCurrency:
    collected_at: datetime | None
    official_last_updated_at: datetime | None


@dataclass(frozen=True)
class GroundedAnswer:
    answer_id: str
    answer: str
    refused: bool
    disclaimer: str | None
    citations: list[Citation]
    data_currency: DataCurrency


def direct_chat_answer(question: str, ui_language: str) -> GroundedAnswer | None:
    language = _normalise_language(ui_language)
    boundary_answer = _boundary_answer(question, language)
    if boundary_answer:
        return boundary_answer

    if not is_greeting(question):
        return None

    if language == "no":
        answer = (
            "Hei! Jeg kan hjelpe med spørsmål om å flytte til Oslo, oppholdstillatelser, "
            "skattekort, ID-nummer, arbeid, bolig, helsetjenester og studentressurser. "
            "Still gjerne et konkret spørsmål, så sjekker jeg de lagrede offentlige kildene."
        )
    else:
        answer = (
            "Hi! I can help with questions about moving to Oslo, residence permits, tax cards, "
            "ID numbers, work, housing, healthcare, and student resources. Ask a concrete question, "
            "and I will check the stored official sources."
        )

    return GroundedAnswer(
        answer_id=str(uuid.uuid4()),
        answer=answer,
        refused=False,
        disclaimer=None,
        citations=[],
        data_currency=DataCurrency(collected_at=None, official_last_updated_at=None),
    )


def is_greeting(question: str) -> bool:
    cleaned = re.sub(r"[!?.\s,]+", " ", question.casefold()).strip()
    if not cleaned:
        return False
    return cleaned in GREETING_TERMS


def _boundary_answer(question: str, language: str) -> GroundedAnswer | None:
    folded = question.casefold()
    if ID_NUMBER_PATTERN.search(question) or any(term in folded for term in PERSONAL_RECORD_TERMS):
        if language == "no":
            answer = (
                "Jeg kan ikke sjekke personlige registre, skatteopplysninger, gjeld eller søknadsstatus. "
                "Bruk den relevante offentlige innloggingstjenesten eller kontakt etaten direkte."
            )
        else:
            answer = (
                "I cannot check personal records, tax records, debt, or application status. "
                "Use the relevant public login service or contact the agency directly."
            )
        return GroundedAnswer(
            answer_id=str(uuid.uuid4()),
            answer=answer,
            refused=True,
            disclaimer=None,
            citations=[],
            data_currency=DataCurrency(collected_at=None, official_last_updated_at=None),
        )

    if any(term in folded for term in LEGAL_DRAFTING_TERMS):
        disclaimer = legal_disclaimer(language)
        if language == "no":
            answer = (
                "Jeg kan ikke skrive klagebrev, fylle ut skjemaer eller lage tekst som skal sendes inn i en "
                "personlig sak. Jeg kan bare forklare generell informasjon fra offentlige kilder."
            )
        else:
            answer = (
                "I cannot write appeal letters, fill forms, or draft text to submit in a personal case. "
                "I can only explain general information from official sources."
            )
        return GroundedAnswer(
            answer_id=str(uuid.uuid4()),
            answer=answer,
            refused=True,
            disclaimer=disclaimer,
            citations=[],
            data_currency=DataCurrency(collected_at=None, official_last_updated_at=None),
        )

    return None


class ChatConfigError(RuntimeError):
    pass


class ChatResponseError(RuntimeError):
    pass


class ChatClient(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> str:
        pass


class OpenAICompatibleChatClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not settings.llm_base_url:
            raise ChatConfigError("LLM_BASE_URL is not configured")
        if not settings.llm_api_key:
            raise ChatConfigError("LLM_API_KEY is not configured")
        if not settings.llm_model:
            raise ChatConfigError("LLM_MODEL is not configured")

        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key.get_secret_value()
        self.model = settings.llm_model
        self._owned_client = client is None
        self.client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owned_client:
            self.client.close()

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        return _extract_chat_content(response.json())


def build_grounded_answer(
    *,
    question: str,
    ui_language: str,
    retrieval: RetrievalResult,
    chat_client: ChatClient,
    session_history: Sequence[ChatMessage] = (),
) -> GroundedAnswer:
    language = _normalise_language(ui_language)
    disclaimer = legal_disclaimer(language) if needs_legal_disclaimer(question) else None

    if retrieval.low_confidence or not retrieval.chunks:
        return _refusal_answer(language=language, disclaimer=disclaimer)

    source_map = _source_map(retrieval.chunks)
    messages = _build_prompt(
        question=question,
        language=language,
        chunks=retrieval.chunks,
        session_history=session_history,
        disclaimer=disclaimer,
    )
    model_text = chat_client.complete(messages)
    parsed = _parse_model_answer(model_text)

    if bool(parsed.get("refusal")):
        return _partial_support_answer(language=language, disclaimer=disclaimer, source_map=source_map)

    raw_answer = str(parsed.get("answer") or "").strip()
    if not raw_answer:
        return _refusal_answer(language=language, disclaimer=disclaimer)

    answer = _expand_grouped_citations(raw_answer)
    answer = _keep_known_citation_markers(answer, source_map)
    answer = _add_missing_citations(answer, default_id="S1")
    if disclaimer:
        answer = _remove_disclaimer_text(answer, disclaimer)
    used_ids = _used_source_ids(answer)
    citations = [source_map[source_id] for source_id in used_ids if source_id in source_map]
    if not citations:
        citations = [source_map["S1"]]

    return GroundedAnswer(
        answer_id=str(uuid.uuid4()),
        answer=answer,
        refused=False,
        disclaimer=disclaimer,
        citations=citations,
        data_currency=_data_currency(citations),
    )


def needs_legal_disclaimer(question: str) -> bool:
    folded = question.casefold()
    return any(term in folded for term in LEGAL_RISK_TERMS)


def legal_disclaimer(language: str) -> str:
    if language == "no":
        return (
            "Dette er generell informasjon fra offentlige kilder, ikke juridisk rådgivning. "
            "Sjekk den relevante etaten eller en kvalifisert rådgiver for din egen sak."
        )
    return (
        "This is general information from official sources, not legal advice. "
        "Check the relevant public agency or a qualified adviser for your own case."
    )


def _refusal_answer(*, language: str, disclaimer: str | None) -> GroundedAnswer:
    if language == "no":
        answer = (
            "Jeg finner ikke nok støtte i de lagrede offentlige kildene til å svare trygt. "
            "Sjekk den relevante offentlige nettsiden direkte."
        )
    else:
        answer = (
            "I do not have enough support in the stored official sources to answer safely. "
            "Please check the relevant public website directly."
        )
    return GroundedAnswer(
        answer_id=str(uuid.uuid4()),
        answer=answer,
        refused=True,
        disclaimer=disclaimer,
        citations=[],
        data_currency=DataCurrency(collected_at=None, official_last_updated_at=None),
    )


def _partial_support_answer(
    *,
    language: str,
    disclaimer: str | None,
    source_map: dict[str, Citation],
) -> GroundedAnswer:
    primary = source_map["S1"]
    if language == "no":
        answer = (
            "De lagrede utdragene oppgir ikke den nøyaktige detaljen i spørsmålet. "
            f"Den mest relevante offentlige kilden å sjekke er {primary.source_owner}, "
            f"seksjonen \"{primary.section_heading}\". [S1]"
        )
    else:
        answer = (
            "The stored excerpts do not give the exact detail in the question. "
            f"The most relevant official source to check is {primary.source_owner}, "
            f"section \"{primary.section_heading}\". [S1]"
        )

    citations = [primary]
    return GroundedAnswer(
        answer_id=str(uuid.uuid4()),
        answer=answer,
        refused=False,
        disclaimer=disclaimer,
        citations=citations,
        data_currency=_data_currency(citations),
    )


def _build_prompt(
    *,
    question: str,
    language: str,
    chunks: Sequence[RetrievedChunk],
    session_history: Sequence[ChatMessage],
    disclaimer: str | None,
) -> list[ChatMessage]:
    language_name = "Norwegian Bokmål" if language == "no" else "English"
    context = _format_context(chunks)
    history = _format_history(session_history)
    disclaimer_rule = (
        f'Do not include this disclaimer in the answer text; it is shown separately by the app: "{disclaimer}".'
        if disclaimer
        else "Do not add a legal disclaimer unless the question asks for personal legal advice."
    )

    system = (
        "You answer for a public demo about moving to Oslo and Norway. "
        "Use only the supplied official source excerpts for factual public-service information. "
        "Keep the language simple, around B1/B2. "
        "Do not decide eligibility, fill forms, or invent missing rules. "
        "Do not say 'sources you provided'; refer to them as stored official sources or excerpts. "
        "For legal-risk questions, answer only the supported general information and tell the user to check "
        "the relevant agency or qualified adviser for their own case. "
        "For follow-up questions, use the session history only to understand what the user refers to; "
        "the factual answer must still come from the source excerpts. "
        "If the answer language differs from the source language, translate only supported details. "
        "If the excerpts partly answer the question, give the supported part instead of refusing. "
        "Every factual sentence about public rules, documents, dates, fees, services, or procedures "
        "must include a source marker like [S1]. "
        "Return only JSON with keys: answer, refusal."
    )
    user = (
        f"Answer language: {language_name}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Session history, if useful for context only:\n{history}\n\n"
        f"Official source excerpts:\n{context}\n\n"
        f"{disclaimer_rule}\n"
        "Set refusal to true only when the excerpts do not support any useful answer to the question."
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    parts = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.text.split())
        block = (
            f"[S{index}]\n"
            f"Owner: {chunk.source_owner}\n"
            f"URL: {chunk.section_url or chunk.source_url}\n"
            f"Section: {chunk.section_heading}\n"
            f"Collected: {chunk.collected_at.date().isoformat()}\n"
            f"Official last updated: {_date_or_unknown(chunk.official_last_updated_at)}\n"
            f"Text: {text}\n"
        )
        if used_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used_chars += len(block)
    return "\n".join(parts)


def _format_history(messages: Sequence[ChatMessage]) -> str:
    clean_messages = [
        ChatMessage(role=message.role, content=" ".join(message.content.split()))
        for message in messages[-6:]
        if message.role in {"user", "assistant"} and message.content.strip()
    ]
    if not clean_messages:
        return "None"
    return "\n".join(f"{message.role}: {message.content[:500]}" for message in clean_messages)


def _extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatResponseError("Chat response did not contain choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text") for part in content if isinstance(part, dict)]
        joined = "".join(part for part in text_parts if isinstance(part, str))
        if joined:
            return joined
    raise ChatResponseError("Chat response did not contain message content")


def _parse_model_json(content: str) -> dict[str, Any]:
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
            raise ChatResponseError("Chat response was not valid JSON") from None
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise ChatResponseError("Chat response JSON was not an object")
    return payload


def _parse_model_answer(content: str) -> dict[str, Any]:
    try:
        return _parse_model_json(content)
    except ChatResponseError:
        clean_answer = content.strip()
        if not clean_answer:
            raise
        return {
            "answer": clean_answer,
            "refusal": _looks_like_refusal(clean_answer),
        }


def _looks_like_refusal(answer: str) -> bool:
    folded = answer.casefold()
    refusal_terms = (
        "not enough support",
        "cannot answer",
        "can't answer",
        "do not have enough",
        "insufficient",
        "finner ikke nok støtte",
        "kan ikke svare",
        "ikke nok grunnlag",
    )
    return any(term in folded for term in refusal_terms)


def _source_map(chunks: Sequence[RetrievedChunk]) -> dict[str, Citation]:
    return {
        f"S{index}": Citation(
            citation_id=f"S{index}",
            chunk_id=chunk.chunk_id,
            source_owner=chunk.source_owner,
            source_url=chunk.source_url,
            section_url=chunk.section_url,
            section_heading=chunk.section_heading,
            collected_at=chunk.collected_at,
            official_last_updated_at=chunk.official_last_updated_at,
        )
        for index, chunk in enumerate(chunks, start=1)
    }


def _keep_known_citation_markers(answer: str, source_map: dict[str, Citation]) -> str:
    return CITATION_PATTERN.sub(
        lambda match: f"[S{match.group(1)}]" if f"S{match.group(1)}" in source_map else "",
        answer,
    )


def _expand_grouped_citations(answer: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source_ids = [part.strip() for part in match.group(1).split(",")]
        return " ".join(f"[{source_id}]" for source_id in source_ids)

    return GROUPED_CITATION_PATTERN.sub(replace, answer)


def _add_missing_citations(answer: str, *, default_id: str) -> str:
    lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or CITATION_PATTERN.search(stripped):
            lines.append(line)
            continue
        if stripped.startswith(("-", "*")):
            lines.append(f"{line} [{default_id}]")
        else:
            lines.append(f"{line.rstrip()} [{default_id}]")
    return "\n".join(lines).strip()


def _remove_disclaimer_text(answer: str, disclaimer: str) -> str:
    return answer.replace(disclaimer, "").strip()


def _used_source_ids(answer: str) -> list[str]:
    seen: list[str] = []
    for match in CITATION_PATTERN.finditer(answer):
        source_id = f"S{match.group(1)}"
        if source_id not in seen:
            seen.append(source_id)
    return seen


def _data_currency(citations: Sequence[Citation]) -> DataCurrency:
    collected_dates = [citation.collected_at for citation in citations]
    updated_dates = [
        citation.official_last_updated_at
        for citation in citations
        if citation.official_last_updated_at is not None
    ]
    return DataCurrency(
        collected_at=max(collected_dates) if collected_dates else None,
        official_last_updated_at=max(updated_dates) if updated_dates else None,
    )


def _date_or_unknown(value: datetime | None) -> str:
    return value.date().isoformat() if value else "unknown"


def _normalise_language(language: str) -> str:
    return "no" if language.lower().startswith("no") else "en"
