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
    return direct_chat_answer(question, ui_language)


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
