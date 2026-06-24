from __future__ import annotations

import re
from collections.abc import Sequence

from sqlalchemy.orm import Session

from oslo_newcomer_rag.generation import (
    ChatMessage,
    GroundedAnswer,
    direct_chat_answer,
    is_general_chat_question,
)
from oslo_newcomer_rag.retrieval import (
    OpenAICompatibleEmbeddingClient,
    RetrievalResult,
    RetrievedChunk,
    expand_retrieval_terms,
    retrieve_chunks_with_language_fallback,
)


# Common Norwegian function words. Combined with the å/æ/ø signal this is enough to
# tell Norwegian from English on short questions; topic nouns are not needed.
NORWEGIAN_WORDS = {
    "jeg", "du", "vi", "de", "det", "den", "dette", "som",
    "og", "eller", "ikke", "har", "kan", "skal", "må", "får",
    "hva", "hvor", "hvordan", "hvem", "hvilke", "når", "hvis",
    "om", "til", "fra", "med", "av", "på", "å",
}
ENGLISH_WORDS = {
    "the", "you", "your", "are", "is", "do", "does", "have",
    "can", "should", "would", "what", "where", "when", "how",
    "who", "which", "and", "or", "if", "to", "from", "with",
    "this", "that", "there", "get", "got", "apply", "about",
}
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
# Referential pronouns plus the few topics newcomers ask follow-ups about.
FOLLOW_UP_TERMS = {
    "it", "that", "this", "there", "they", "them",
    "det", "den", "dette", "her",
    "documents", "dokumenter", "fees", "cost", "money", "penger",
    "appointment", "office", "kontor", "waiting", "venteliste", "ventelisten",
}
REFUSAL_MARKERS = (
    "i do not have enough support",
    "could not answer safely",
    "jeg finner ikke nok støtte",
    "kunne ikke svare trygt",
)
MAX_CONTEXT_MESSAGE_CHARS = 360
TOPIC_QUERY_HINTS = (
    (
        ("family immigration", "family reunification", "familieinnvandring", "familiegjenforening"),
        "The applicant is the person who wishes to visit or live in Norway. "
        "Family immigration is also called family reunification. "
        "Spouse cohabitant child UDI family immigration",
        ("https://www.udi.no/en/want-to-apply/family-immigration/",),
    ),
    (
        ("citizenship", "norwegian citizen", "statsborgerskap", "norsk statsborger"),
        "The applicant is the person who wishes to become a Norwegian citizen. "
        "Norwegian citizenship citizenship rules UDI",
        ("https://www.udi.no/en/want-to-apply/citizenship/",),
    ),
    (
        ("permanent residence", "permanent residence permit", "permanent opphold", "permanent oppholdstillatelse"),
        "Apply for a permanent residence permit. "
        "Permanent residence permit eligible requirements Norwegian language social studies UDI",
        ("https://www.udi.no/en/want-to-apply/permanent-residence/permanent-residence-permit/",),
    ),
    (
        ("residence card", "residence cards", "oppholdskort"),
        "Residence card proves that you have been granted a residence permit in Norway. "
        "Residence card UDI police appointment EU EEA family members",
        ("https://www.udi.no/en/word-definitions/-residence-cards/",),
    ),
)


def build_boundary_answer(question: str, ui_language: str) -> GroundedAnswer | None:
    return direct_chat_answer(question, infer_answer_language(question, ui_language))


def should_use_general_chat(question: str) -> bool:
    return is_general_chat_question(question)


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


def build_retrieval_query(
    question: str,
    session_history: Sequence[ChatMessage],
    *,
    planned_query: str | None = None,
) -> str:
    return build_retrieval_queries(question, session_history, planned_query=planned_query)[-1]


def build_retrieval_queries(
    question: str,
    session_history: Sequence[ChatMessage],
    *,
    planned_query: str | None = None,
) -> list[str]:
    clean_question = " ".join(question.split())
    search_question = " ".join((planned_query or clean_question).split())
    context = _conversation_context(session_history)
    queries = [expand_retrieval_terms(clean_question)]
    expanded_question = expand_retrieval_terms(search_question)
    if expanded_question.casefold() != queries[0].casefold():
        queries.append(expanded_question)

    if context and _looks_like_follow_up(search_question):
        contextual = expand_retrieval_terms(f"{context} {search_question}")
        if contextual != expanded_question:
            queries.append(contextual)

    topic_text = f"{context} {clean_question} {search_question}".strip()
    queries.extend(_topic_hint_queries(topic_text))
    return _dedupe_queries(queries)


def retrieve_for_chat(
    session: Session,
    embedder: OpenAICompatibleEmbeddingClient,
    *,
    question: str,
    ui_language: str,
    session_history: Sequence[ChatMessage],
    planned_query: str | None = None,
    log_query: bool = True,
) -> RetrievalResult:
    queries = build_retrieval_queries(question, session_history, planned_query=planned_query)
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
    merged = _merge_retrieval_results(results)
    return _focus_topic_source_chunks(merged, planned_query or question)


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

    if no_score > en_score:
        return "no"
    if en_score > no_score:
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
    return any(word in FOLLOW_UP_TERMS for word in words)


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


def _topic_hint_queries(text: str) -> list[str]:
    folded = text.casefold()
    return [query for triggers, query, _ in TOPIC_QUERY_HINTS if any(trigger in folded for trigger in triggers)]


def _topic_source_urls(text: str) -> tuple[str, ...]:
    folded = text.casefold()
    if any(trigger in folded for trigger in TOPIC_QUERY_HINTS[1][0]):
        return TOPIC_QUERY_HINTS[1][2]

    urls: list[str] = []
    for triggers, _, source_urls in TOPIC_QUERY_HINTS:
        if any(trigger in folded for trigger in triggers):
            urls.extend(source_urls)
    return tuple(dict.fromkeys(urls))


def _dedupe_queries(queries: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean_query = " ".join(query.split())
        key = clean_query.casefold()
        if not clean_query or key in seen:
            continue
        deduped.append(clean_query)
        seen.add(key)
    return deduped


def _focus_topic_source_chunks(result: RetrievalResult, question: str) -> RetrievalResult:
    source_urls = _topic_source_urls(question)
    if not source_urls or not result.chunks:
        return result

    focused = [chunk for chunk in result.chunks if chunk.source_url in source_urls]
    if not focused:
        return result

    return RetrievalResult(
        query=result.query,
        chunks=focused[:6],
        low_confidence=False,
        language_fallback_used=result.language_fallback_used,
    )


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
