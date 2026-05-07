from datetime import UTC, datetime

from oslo_newcomer_rag.chat_flow import (
    _focus_topic_source_chunks,
    build_boundary_answer,
    build_retrieval_queries,
    build_retrieval_query,
    infer_answer_language,
    should_use_general_chat,
)
from oslo_newcomer_rag.generation import ChatMessage
from oslo_newcomer_rag.retrieval import RetrievalResult, RetrievedChunk


def test_answer_language_follows_current_message_not_ui_toggle() -> None:
    assert infer_answer_language("I just got a job in Oslo. How do I get a D-number?", "no") == "en"
    assert (
        infer_answer_language("What are the best cheap bars in Grünerløkka for international students?", "no")
        == "en"
    )
    assert (
        infer_answer_language(
            "Hei! Jeg er en internasjonal student. Hvordan søker jeg om studietillatelse?",
            "en",
        )
        == "no"
    )


def test_ambiguous_answer_language_can_fall_back_to_recent_user_message() -> None:
    language = infer_answer_language(
        "Next?",
        "no",
        [ChatMessage(role="user", content="How do I apply for student housing through SiO?")],
    )

    assert language == "en"


def test_greeting_uses_general_chat_instead_of_boundary_answer() -> None:
    answer = build_boundary_answer("hi", "no")

    assert answer is None
    assert should_use_general_chat("hi") is True
    assert should_use_general_chat("what can you do?") is True


def test_short_follow_up_uses_recent_user_question_for_retrieval() -> None:
    query = build_retrieval_query(
        "Anywhere else?",
        [
            ChatMessage(role="user", content="Where can students find housing support?"),
            ChatMessage(role="assistant", content="Students can check SiO housing pages."),
        ],
    )

    assert "Where can students find housing support?" in query
    assert "Anywhere else?" in query
    assert "student services" in query


def test_norwegian_follow_up_gets_context_and_glossary_terms() -> None:
    query = build_retrieval_query(
        "Hva med skattekort?",
        [ChatMessage(role="user", content="Jeg skal jobbe i Oslo.")],
    )

    assert "Jeg skal jobbe i Oslo." in query
    assert "Hva med skattekort?" in query
    assert "tax deduction card" in query


def test_d_number_appointment_follow_up_keeps_previous_topic() -> None:
    queries = build_retrieval_queries(
        "Do I have to book an appointment to get it?",
        [
            ChatMessage(role="user", content="I just got a job in Oslo. How do I get a D-number?"),
            ChatMessage(role="assistant", content="D numbers depend on your residence permit. [S1]"),
        ],
    )

    assert len(queries) == 2
    assert "D-number" in queries[-1]
    assert "book appointment" in queries[-1]


def test_sua_documents_follow_up_ignores_previous_refusal() -> None:
    query = build_retrieval_query(
        "Can I go to the SUA office for that? What should I bring?",
        [
            ChatMessage(role="user", content="I just got a job in Oslo. How do I get a D-number?"),
            ChatMessage(role="assistant", content="D numbers depend on your residence permit. [S1]"),
            ChatMessage(role="user", content="Do I have to book an appointment to get it?"),
            ChatMessage(role="assistant", content="I do not have enough support in the stored official sources."),
        ],
    )

    assert "SUA" in query
    assert "D-number" in query
    assert "not have enough support" not in query


def test_study_permit_money_and_work_follow_ups_keep_context() -> None:
    history = [
        ChatMessage(role="user", content="How do I apply for a study permit in Norway?"),
        ChatMessage(role="assistant", content="You can apply for a study permit through UDI. [S1]"),
    ]

    money_query = build_retrieval_query("Hvor mye penger må jeg bevise at jeg har?", history)
    work_query = build_retrieval_query("Kan jeg jobbe deltid mens jeg studerer her?", history)

    assert "study permit" in money_query
    assert "proof of funds" in money_query
    assert "part-time work" in work_query


def test_housing_waiting_list_and_pet_follow_ups_keep_context() -> None:
    history = [
        ChatMessage(role="user", content="How do I apply for student housing through SiO?"),
        ChatMessage(role="assistant", content="You can apply through SiO student housing. [S1]"),
    ]

    wait_query = build_retrieval_query("Hvor lang er ventelisten?", history)
    pet_query = build_retrieval_query("Er det lov å ha med hund eller katt i leiligheten?", history)

    assert "student housing" in wait_query
    assert "waiting list" in wait_query
    assert "rental" in pet_query


def test_dagpenger_follow_up_keeps_nav_context() -> None:
    query = build_retrieval_query(
        "What if I get a job, but then lose it after 3 months? Can I get dagpenger then?",
        [
            ChatMessage(role="user", content="Can EU citizens get unemployment benefits in Norway?"),
            ChatMessage(role="assistant", content="NAV explains jobseeker registration and dagpenger. [S1]"),
        ],
    )

    assert "unemployment benefits" in query
    assert "jobseeker" in query


def test_source_navigation_examples_get_topic_hint_queries() -> None:
    family_queries = build_retrieval_queries("Where can I read about family immigration?", [])
    citizenship_queries = build_retrieval_queries("Where can I check citizenship rules?", [])
    permanent_queries = build_retrieval_queries("Where can I check information about permanent residence?", [])
    typo_queries = build_retrieval_queries(
        "what do I need to apply permananet residency",
        [],
        planned_query="permanent residence permit application requirements UDI",
    )
    citizenship_typo_queries = build_retrieval_queries(
        "I have permanent residency, what do I need to apply to cetezenship?",
        [],
        planned_query="Norwegian citizenship application requirements for a person with permanent residence UDI",
    )

    assert len(family_queries) == 2
    assert "Family immigration is also called family reunification" in family_queries[-1]
    assert len(citizenship_queries) == 2
    assert "become a Norwegian citizen" in citizenship_queries[-1]
    assert len(permanent_queries) == 2
    assert "permanent residence permit" in permanent_queries[-1]
    assert any("permanent residence" in query for query in typo_queries)
    assert "permanent residence permit" in typo_queries[-1]
    assert any("citizenship" in query for query in citizenship_typo_queries)
    assert any("become a Norwegian citizen" in query for query in citizenship_typo_queries)


def test_source_navigation_examples_keep_matching_source_chunks() -> None:
    result = RetrievalResult(
        query="citizenship topic hint",
        chunks=[
            _chunk("https://www.udi.no/en/want-to-apply/citizenship/", "Citizenship", 0.49),
            _chunk("https://www.udi.no/en/want-to-apply/work-immigration/", "Work immigration", 0.47),
        ],
        low_confidence=False,
    )

    focused = _focus_topic_source_chunks(result, "Where can I check citizenship rules?")

    assert [chunk.source_url for chunk in focused.chunks] == [
        "https://www.udi.no/en/want-to-apply/citizenship/"
    ]


def test_source_focus_handles_repaired_topic_wording() -> None:
    result = RetrievalResult(
        query="permanent residence topic hint",
        chunks=[
            _chunk(
                "https://www.udi.no/en/want-to-apply/permanent-residence/permanent-residence-permit/",
                "Permanent residence",
                0.49,
            ),
            _chunk(
                "https://www.skatteetaten.no/en/person/national-registry/moving/",
                "National registry",
                0.47,
            ),
        ],
        low_confidence=False,
    )

    focused = _focus_topic_source_chunks(
        result,
        "what do I need to apply permanent residence",
    )

    assert [chunk.source_url for chunk in focused.chunks] == [
        "https://www.udi.no/en/want-to-apply/permanent-residence/permanent-residence-permit/"
    ]


def test_citizenship_source_focus_treats_permanent_residence_as_context() -> None:
    result = RetrievalResult(
        query="citizenship and permanent residence",
        chunks=[
            _chunk("https://www.udi.no/en/want-to-apply/citizenship/", "Citizenship", 0.48),
            _chunk(
                "https://www.udi.no/en/want-to-apply/permanent-residence/permanent-residence-permit/",
                "Permanent residence",
                0.47,
            ),
        ],
        low_confidence=False,
    )

    focused = _focus_topic_source_chunks(
        result,
        "I have permanent residence, what do I need to apply to citizenship?",
    )

    assert [chunk.source_url for chunk in focused.chunks] == [
        "https://www.udi.no/en/want-to-apply/citizenship/"
    ]


def _chunk(source_url: str, heading: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{heading}",
        source_id="source-1",
        source_owner="UDI",
        source_url=source_url,
        category="immigration",
        language="en",
        section_heading=heading,
        section_url=source_url,
        text=heading,
        collected_at=datetime(2026, 5, 3, tzinfo=UTC),
        official_last_updated_at=None,
        score=score,
        vector_score=score,
        keyword_score=0.0,
    )
