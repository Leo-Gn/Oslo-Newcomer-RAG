from __future__ import annotations

import re
from collections.abc import Sequence

from sqlalchemy.orm import Session

from oslo_newcomer_rag.generation import ChatMessage, direct_chat_answer
from oslo_newcomer_rag.retrieval import (
    OpenAICompatibleEmbeddingClient,
    RetrievalResult,
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
    "else",
    "documents",
    "fees",
    "cost",
    "costs",
    "deadline",
    "deadlines",
    "appointment",
    "annet",
    "dokumenter",
    "gebyr",
    "frist",
    "time",
}


def build_direct_answer(question: str, ui_language: str):
    return direct_chat_answer(question, ui_language)


def build_retrieval_query(question: str, session_history: Sequence[ChatMessage]) -> str:
    clean_question = " ".join(question.split())
    recent_user_questions = [
        " ".join(message.content.split())
        for message in session_history
        if message.role == "user" and message.content.strip()
    ][-2:]

    if not recent_user_questions or not _looks_like_follow_up(clean_question):
        return expand_retrieval_terms(clean_question)

    return expand_retrieval_terms(" ".join([*recent_user_questions, clean_question]))


def retrieve_for_chat(
    session: Session,
    embedder: OpenAICompatibleEmbeddingClient,
    *,
    question: str,
    ui_language: str,
    session_history: Sequence[ChatMessage],
    log_query: bool = True,
) -> RetrievalResult:
    retrieval_query = build_retrieval_query(question, session_history)
    return retrieve_chunks_with_language_fallback(
        session,
        embedder,
        retrieval_query,
        preferred_language=ui_language,
        log_query=log_query,
    )


def _looks_like_follow_up(question: str) -> bool:
    folded = question.casefold().strip()
    if not folded:
        return False
    if any(folded.startswith(prefix) for prefix in FOLLOW_UP_STARTS):
        return True

    words = re.findall(r"[\wøæåØÆÅ-]+", folded)
    if len(words) <= 5 and any(term in words for term in FOLLOW_UP_TERMS):
        return True
    return False
