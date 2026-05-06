from __future__ import annotations

import re
from collections.abc import Sequence

from sqlalchemy.orm import Session

from oslo_newcomer_rag.generation import ChatMessage, direct_chat_answer
from oslo_newcomer_rag.retrieval import (
    OpenAICompatibleEmbeddingClient,
    RetrievalResult,
    RetrievedChunk,
    expand_retrieval_terms,
    retrieve_chunks_with_language_fallback,
)


NORWEGIAN_WORDS = {
    "at",
    "av",
    "bare",
    "bli",
    "bolig",
    "den",
    "der",
    "det",
    "dette",
    "ditt",
    "du",
    "eller",
    "etter",
    "fastlege",
    "fikk",
    "for",
    "fra",
    "få",
    "får",
    "hallo",
    "har",
    "hei",
    "her",
    "hva",
    "hvem",
    "hvilke",
    "hvis",
    "hvor",
    "hvordan",
    "ikke",
    "jeg",
    "jobb",
    "jobbe",
    "kan",
    "kommune",
    "leiligheten",
    "med",
    "meg",
    "mens",
    "må",
    "norge",
    "norsk",
    "norskkurs",
    "om",
    "oppholdstillatelse",
    "penger",
    "på",
    "skal",
    "skattekort",
    "som",
    "statsborgerskap",
    "studere",
    "studerer",
    "søke",
    "søker",
    "tillatelse",
    "tillatelsen",
    "til",
    "venteliste",
    "ventelisten",
    "å",
}
ENGLISH_WORDS = {
    "after",
    "am",
    "and",
    "any",
    "apply",
    "appointment",
    "are",
    "before",
    "bars",
    "best",
    "book",
    "bring",
    "can",
    "card",
    "check",
    "cheap",
    "do",
    "does",
    "for",
    "from",
    "get",
    "got",
    "have",
    "hello",
    "hi",
    "how",
    "housing",
    "if",
    "international",
    "job",
    "letter",
    "money",
    "moving",
    "my",
    "number",
    "office",
    "owe",
    "permit",
    "personal",
    "received",
    "records",
    "residence",
    "saying",
    "should",
    "student",
    "students",
    "study",
    "tax",
    "that",
    "the",
    "there",
    "this",
    "to",
    "what",
    "where",
    "work",
    "write",
    "you",
}
NORWEGIAN_PHRASES = (
    "hva med",
    "hvor kan",
    "hvordan får",
    "kan jeg",
    "må jeg",
    "jeg skal",
    "jeg har",
    "oppholdstillatelse",
)
ENGLISH_PHRASES = (
    "how do",
    "where can",
    "do i",
    "can i",
    "should i",
    "i have",
    "i just",
    "tax card",
    "residence permit",
)
NORWEGIAN_GREETING_WORDS = {"hei", "heisann", "hallo"}
ENGLISH_GREETING_WORDS = {"hi", "hello", "hey"}
FOLLOW_UP_STARTS = (
    "what about",
    "what else",
    "where else",
    "anywhere else",
    "anything else",
    "and ",
    "also ",
    "how about",
    "what documents",
    "which documents",
    "hva med",
    "hvor ellers",
    "noe annet",
    "hvilke dokumenter",
    "og ",
)
FOLLOW_UP_TERMS = {
    "it",
    "that",
    "this",
    "there",
    "here",
    "they",
    "them",
    "office",
    "bring",
    "else",
    "documents",
    "fees",
    "cost",
    "costs",
    "money",
    "funds",
    "deadline",
    "deadlines",
    "appointment",
    "work",
    "study",
    "studying",
    "waitlist",
    "waiting",
    "list",
    "det",
    "den",
    "dette",
    "her",
    "dit",
    "kontor",
    "ta",
    "bringe",
    "annet",
    "dokumenter",
    "penger",
    "gebyr",
    "frist",
    "time",
    "jobbe",
    "studere",
    "studerer",
    "tillatelsen",
    "ventelisten",
    "venteliste",
}
REFUSAL_MARKERS = (
    "i do not have enough support",
    "could not answer safely",
    "jeg finner ikke nok støtte",
    "kunne ikke svare trygt",
)
MAX_CONTEXT_MESSAGE_CHARS = 360


def build_direct_answer(question: str, ui_language: str):
    return direct_chat_answer(question, infer_answer_language(question, ui_language))


def infer_answer_language(
    question: str,
    ui_language: str,
    session_history: Sequence[ChatMessage] = (),
) -> str:
    clean_question = " ".join(question.split())
    if not clean_question:
        return _normalise_language(ui_language)

    language = _detect_language(clean_question)
    if language:
        return language

    for message in reversed(session_history):
        if message.role != "user":
            continue
        language = _detect_language(message.content)
        if language:
            return language

    return _normalise_language(ui_language)


def build_retrieval_query(question: str, session_history: Sequence[ChatMessage]) -> str:
    return build_retrieval_queries(question, session_history)[-1]


def build_retrieval_queries(question: str, session_history: Sequence[ChatMessage]) -> list[str]:
    clean_question = " ".join(question.split())
    context = _conversation_context(session_history)
    expanded_question = expand_retrieval_terms(clean_question)

    if not context or not _looks_like_follow_up(clean_question):
        return [expanded_question]

    contextual = expand_retrieval_terms(f"{context} {clean_question}")
    if contextual == expanded_question:
        return [expanded_question]
    return [expanded_question, contextual]


def retrieve_for_chat(
    session: Session,
    embedder: OpenAICompatibleEmbeddingClient,
    *,
    question: str,
    ui_language: str,
    session_history: Sequence[ChatMessage],
    log_query: bool = True,
) -> RetrievalResult:
    queries = build_retrieval_queries(question, session_history)
    results = [
        retrieve_chunks_with_language_fallback(
            session,
            embedder,
            query,
            preferred_language=ui_language,
            log_query=log_query and index == len(queries) - 1,
        )
        for index, query in enumerate(queries)
    ]
    return _merge_retrieval_results(results)


def _detect_language(text: str) -> str | None:
    folded = text.casefold()
    greeting = re.sub(r"[!?.\s,]+", " ", folded).strip()
    if greeting in NORWEGIAN_GREETING_WORDS:
        return "no"
    if greeting in ENGLISH_GREETING_WORDS:
        return "en"

    words = re.findall(r"[\wøæåØÆÅ-]+", folded)
    if not words:
        return None

    no_score = sum(1 for word in words if word in NORWEGIAN_WORDS)
    en_score = sum(1 for word in words if word in ENGLISH_WORDS)
    if re.search(r"\bI\b", text):
        en_score += 1
    if re.search(r"[æøåÆØÅ]", text):
        no_score += 2
    no_score += sum(2 for phrase in NORWEGIAN_PHRASES if phrase in folded)
    en_score += sum(2 for phrase in ENGLISH_PHRASES if phrase in folded)

    if no_score >= en_score + 2:
        return "no"
    if en_score >= no_score + 2:
        return "en"
    return None


def _normalise_language(language: str) -> str:
    return "no" if language.lower().startswith("no") else "en"


def _looks_like_follow_up(question: str) -> bool:
    folded = question.casefold().strip()
    if not folded:
        return False
    if any(folded.startswith(prefix) for prefix in FOLLOW_UP_STARTS):
        return True

    words = re.findall(r"[\wøæåØÆÅ-]+", folded)
    if any(term in words for term in FOLLOW_UP_TERMS):
        return True
    if len(words) <= 8 and any(term in folded for term in FOLLOW_UP_TERMS):
        return True
    return False


def _conversation_context(messages: Sequence[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages[-8:]:
        content = " ".join(message.content.split())
        if not content or _is_generic_refusal(content):
            continue
        if message.role == "user":
            parts.append(content[:MAX_CONTEXT_MESSAGE_CHARS])
        elif message.role == "assistant":
            parts.append(_strip_citations(content)[:MAX_CONTEXT_MESSAGE_CHARS])
    return " ".join(parts[-4:])


def _is_generic_refusal(content: str) -> bool:
    folded = content.casefold()
    return any(marker in folded for marker in REFUSAL_MARKERS)


def _strip_citations(content: str) -> str:
    return re.sub(r"\[S\d+\]", "", content).strip()


def _merge_retrieval_results(results: Sequence[RetrievalResult]) -> RetrievalResult:
    if not results:
        return RetrievalResult(query="", chunks=[], low_confidence=True)

    chunks: dict[str, RetrievedChunk] = {}
    for result in results:
        for chunk in result.chunks:
            existing = chunks.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                chunks[chunk.chunk_id] = chunk

    merged_chunks = sorted(chunks.values(), key=lambda chunk: chunk.score, reverse=True)[:6]
    return RetrievalResult(
        query=results[-1].query,
        chunks=merged_chunks,
        low_confidence=not merged_chunks,
        language_fallback_used=any(result.language_fallback_used for result in results),
    )
