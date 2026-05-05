from oslo_newcomer_rag.chat_flow import build_retrieval_query
from oslo_newcomer_rag.generation import ChatMessage


def test_short_follow_up_uses_recent_user_question_for_retrieval() -> None:
    query = build_retrieval_query(
        "Anywhere else?",
        [
            ChatMessage(role="user", content="Where can students find housing support?"),
            ChatMessage(role="assistant", content="Students can check SiO housing pages."),
        ],
    )

    assert query.startswith("Where can students find housing support? Anywhere else?")
    assert "student services" in query


def test_norwegian_follow_up_gets_context_and_glossary_terms() -> None:
    query = build_retrieval_query(
        "Hva med skattekort?",
        [ChatMessage(role="user", content="Jeg skal jobbe i Oslo.")],
    )

    assert "Jeg skal jobbe i Oslo." in query
    assert "Hva med skattekort?" in query
    assert "tax deduction card" in query
