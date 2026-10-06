from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.generation.answer_generator import AnswerGenerator
from backend.schemas import DocumentChunk, FilingMetadata, FilingSectionName, RetrievedChunk


@pytest.fixture
def generator_and_completion(monkeypatch):
    completion = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer [1]."))]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(
        "backend.generation.answer_generator.OpenAI", Mock(return_value=client)
    )
    return AnswerGenerator(), completion


def make_result(
    text: str,
    *,
    rank: int = 1,
    section: FilingSectionName = "Item 7",
    section_title: str = "Management's Discussion and Analysis",
) -> RetrievedChunk:
    metadata = FilingMetadata(
        company="MICROSOFT CORP",
        ticker="MSFT",
        cik="789019",
        accession_number="0000950170-23-035122",
        filing_type="10-K",
        fiscal_year=2023,
        filing_date="2023-07-27",
        source_url="https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
    )
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"chunk-{rank}",
            metadata=metadata,
            section=section,
            section_title=section_title,
            text=text,
            char_start=0,
            char_end=len(text),
            token_count=max(1, len(text) // 4),
        ),
        score=1.0 / rank,
        rank=rank,
        retrieval_method="bm25",
    )


def test_generation_receives_evidence_beyond_citation_preview(generator_and_completion) -> None:
    generator, completion = generator_and_completion
    evidence = "Cash, cash equivalents, and short-term investments totaled $111.3 billion."
    text = "Background filing disclosure. " * 50 + "\n" + evidence
    result = make_result(text)

    answer = generator.generate_answer("What was Microsoft's cash balance?", [result])

    completion.assert_called_once()
    prompt = completion.call_args.kwargs["messages"][1]["content"]
    assert text.index(evidence) > 1_200
    assert text in prompt
    assert evidence in prompt
    assert evidence not in answer.citations[0].excerpt
    assert answer.citations[0].excerpt.endswith("...")
    assert len(answer.citations[0].excerpt) <= 1_203
    assert answer.retrieved_chunks == [result]
    assert answer.answer == "Grounded answer [1]."


def test_generation_preserves_chunk_order_and_citation_headers(generator_and_completion) -> None:
    generator, completion = generator_and_completion
    first = make_result("Operating cash flow\nCash from operations totaled $87.6 billion.")
    second = make_result(
        "Revenue recognition\nSubscription revenue is recognized over the contract period.",
        rank=2,
        section="Item 8",
        section_title="Financial Statements",
    )

    answer = generator.generate_answer("Summarize the two disclosures.", [first, second])

    completion.assert_called_once()
    prompt = completion.call_args.kwargs["messages"][1]["content"]
    first_header = "[1] MICROSOFT CORP 2023 10-K, Item 7 - Management's Discussion and Analysis"
    second_header = "[2] MICROSOFT CORP 2023 10-K, Item 8 - Financial Statements"
    assert first_header + "\n" + first.chunk.text in prompt
    assert second_header + "\n" + second.chunk.text in prompt
    assert prompt.index(first_header) < prompt.index(second_header)
    assert [(citation.citation_id, citation.chunk_id) for citation in answer.citations] == [
        (1, first.chunk.chunk_id),
        (2, second.chunk.chunk_id),
    ]
    assert answer.retrieved_chunks == [first, second]


def test_empty_retrieval_returns_limitation_without_llm_call(generator_and_completion) -> None:
    generator, completion = generator_and_completion

    answer = generator.generate_answer("What was Microsoft's cash balance?", [])

    completion.assert_not_called()
    assert "could not find enough relevant SEC filing context" in answer.answer
    assert answer.citations == []
    assert answer.retrieved_chunks == []
    assert answer.limitations == ["No relevant chunks were retrieved."]
