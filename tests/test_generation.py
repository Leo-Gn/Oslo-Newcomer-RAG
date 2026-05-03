from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

import httpx

from oslo_newcomer_rag.config import Settings
from oslo_newcomer_rag.generation import (
    ChatMessage,
    OpenAICompatibleChatClient,
    build_grounded_answer,
    needs_legal_disclaimer,
)
from oslo_newcomer_rag.retrieval import RetrievedChunk, RetrievalResult


class StubChatClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0
        self.messages: list[ChatMessage] = []

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls += 1
        self.messages = list(messages)
        return json.dumps(self.payload)


def test_supported_answer_includes_citations_and_data_currency() -> None:
    chat_client = StubChatClient(
        {
            "answer": "You normally need to register your move and follow the listed steps. [S1]",
            "refusal": False,
        }
    )

    answer = build_grounded_answer(
        question="What should I do after moving to Oslo?",
        ui_language="en",
        retrieval=_retrieval([_chunk()]),
        chat_client=chat_client,
    )

    assert answer.refused is False
    assert answer.answer.endswith("[S1]")
    assert chat_client.calls == 1
    assert len(answer.citations) == 1
    assert answer.citations[0].source_url == "https://www.udi.no/en/"
    assert answer.citations[0].section_heading == "Moving to Norway"
    assert answer.data_currency.collected_at == datetime(2026, 2, 1, tzinfo=UTC)
    assert answer.data_currency.official_last_updated_at == datetime(2026, 1, 20, tzinfo=UTC)


def test_chat_client_uses_openai_compatible_chat_completion_endpoint() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"answer": "Use the official page. [S1]", "refusal": False}
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleChatClient(
        Settings(
            llm_base_url="https://provider.example/v1",
            llm_api_key="test-key",
            llm_model="test-chat",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    content = client.complete([ChatMessage(role="user", content="Hei")])

    assert json.loads(content)["refusal"] is False
    assert seen_payloads[0]["model"] == "test-chat"
    assert seen_payloads[0]["messages"][0] == {"role": "user", "content": "Hei"}


def test_unsupported_question_is_refused_without_calling_model() -> None:
    chat_client = StubChatClient({"answer": "This should not be used.", "refusal": False})

    answer = build_grounded_answer(
        question="Where can I buy a used sofa in Oslo?",
        ui_language="en",
        retrieval=RetrievalResult(query="sofa", chunks=[], low_confidence=True),
        chat_client=chat_client,
    )

    assert answer.refused is True
    assert "not have enough support" in answer.answer
    assert answer.citations == []
    assert answer.data_currency.collected_at is None
    assert chat_client.calls == 0


def test_personal_legal_risk_question_gets_disclaimer() -> None:
    chat_client = StubChatClient(
        {
            "answer": "UDI explains that rejected applications have their own appeal route. [S1]",
            "refusal": False,
        }
    )

    answer = build_grounded_answer(
        question="My application was rejected. Should I appeal?",
        ui_language="en",
        retrieval=_retrieval([_chunk(section_heading="Appeals")]),
        chat_client=chat_client,
    )

    assert needs_legal_disclaimer("My application was rejected. Should I appeal?") is True
    assert answer.disclaimer is not None
    assert "not legal advice" in answer.disclaimer
    assert answer.answer.endswith(answer.disclaimer)


def test_uncited_generated_answer_gets_source_marker() -> None:
    chat_client = StubChatClient(
        {
            "answer": "You should use the official page for the current procedure.",
            "refusal": False,
        }
    )

    answer = build_grounded_answer(
        question="Where do I check the procedure?",
        ui_language="en",
        retrieval=_retrieval([_chunk()]),
        chat_client=chat_client,
    )

    assert answer.answer == "You should use the official page for the current procedure. [S1]"
    assert answer.citations[0].citation_id == "S1"


def test_plain_model_text_is_accepted_and_cited() -> None:
    class PlainTextChatClient:
        def complete(self, messages: Sequence[ChatMessage]) -> str:
            return "Use the official page for the current procedure."

    answer = build_grounded_answer(
        question="Where do I check the procedure?",
        ui_language="en",
        retrieval=_retrieval([_chunk()]),
        chat_client=PlainTextChatClient(),
    )

    assert answer.refused is False
    assert answer.answer == "Use the official page for the current procedure. [S1]"
    assert answer.citations[0].section_url == "https://www.udi.no/en/#moving"


def test_grouped_citation_markers_are_returned_as_separate_citations() -> None:
    chat_client = StubChatClient(
        {
            "answer": "Oslo kommune has newcomer pages about practical steps. [S1, S2]",
            "refusal": False,
        }
    )

    answer = build_grounded_answer(
        question="What should I check after moving?",
        ui_language="en",
        retrieval=_retrieval([_chunk(), _chunk(section_heading="Healthcare")]),
        chat_client=chat_client,
    )

    assert "[S1] [S2]" in answer.answer
    assert [citation.citation_id for citation in answer.citations] == ["S1", "S2"]


def test_norwegian_refusal_uses_requested_language() -> None:
    answer = build_grounded_answer(
        question="Kan jeg få oppholdstillatelse hvis jeg forklarer saken min?",
        ui_language="no",
        retrieval=RetrievalResult(query="oppholdstillatelse", chunks=[], low_confidence=True),
        chat_client=StubChatClient({"answer": "", "refusal": False}),
    )

    assert answer.refused is True
    assert "Jeg finner ikke nok støtte" in answer.answer
    assert "juridisk rådgivning" in answer.answer


def _retrieval(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(query="moving", chunks=chunks, low_confidence=False)


def _chunk(section_heading: str = "Moving to Norway") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="6cbe1e02-30d3-481e-a395-10b287d589f8",
        source_id="9636fd2d-2e0f-48f4-af7b-2893c58b947a",
        source_owner="UDI",
        source_url="https://www.udi.no/en/",
        category="permits",
        language="en",
        section_heading=section_heading,
        section_url="https://www.udi.no/en/#moving",
        text="You must use official information when checking immigration procedures.",
        collected_at=datetime(2026, 2, 1, tzinfo=UTC),
        official_last_updated_at=datetime(2026, 1, 20, tzinfo=UTC),
        score=0.91,
        vector_score=0.82,
        keyword_score=1.0,
    )
