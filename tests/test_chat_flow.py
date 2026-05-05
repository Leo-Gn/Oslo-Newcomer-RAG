from oslo_newcomer_rag.chat_flow import build_retrieval_queries, build_retrieval_query
from oslo_newcomer_rag.generation import ChatMessage


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
