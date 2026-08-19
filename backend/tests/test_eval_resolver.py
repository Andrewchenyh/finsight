import json

import pytest

from backend.chunking.chunker import chunk_filing_sections
from backend.evals.resolver import ResolutionError, resolve_eval_artifact, sha256_bytes
from backend.parsing.section_extractor import extract_filing_sections
from backend.schemas import FilingMetadata, RawFiling


EVIDENCE_QUOTE = (
    "Cyberthreats are constantly evolving and becoming increasingly "
    "sophisticated and difficult to detect."
)
OPTIONAL_EVIDENCE_QUOTE = (
    "Customer security limitations may increase vulnerability to attack."
)


def build_resolver_inputs(
    *,
    quote_occurrences: int = 1,
    review_status: str = "verified",
    include_optional_fact: bool = False,
) -> dict[str, bytes | str]:
    metadata = FilingMetadata(
        company="MICROSOFT CORP",
        ticker="MSFT",
        cik="789019",
        accession_number="0000950170-23-035122",
        filing_type="10-K",
        fiscal_year=2023,
        filing_date="2023-07-27",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/789019/"
            "000095017023035122/test.htm"
        ),
    )
    repeated_context = "Risk discussion continues. " * 45
    quoted_text = " ".join([EVIDENCE_QUOTE] * quote_occurrences)
    raw_content = (
        "<html><body>"
        "<div>Item 1A</div>"
        f"<p>{repeated_context}{quoted_text} {repeated_context} "
        f"{OPTIONAL_EVIDENCE_QUOTE}</p>"
        "<div>Item 7</div>"
        "<p>Management discussion.</p>"
        "</body></html>"
    )
    raw_bytes = raw_content.encode("utf-8")
    raw_filing = RawFiling(
        metadata=metadata,
        content=raw_content,
        content_type="html",
    )
    sections = extract_filing_sections(
        raw_filing,
        target_sections=("Item 1A",),
    )
    chunks = chunk_filing_sections(sections)
    chunks_bytes = json.dumps(
        [chunk.model_dump(mode="json") for chunk in chunks],
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    gold = {
        "schema_version": "1.0",
        "dataset_name": "msft_2023_retrieval",
        "filing": {
            "ticker": "MSFT",
            "fiscal_year": 2023,
            "filing_type": "10-K",
            "accession_number": "0000950170-23-035122",
        },
        "questions": [
            {
                "id": "msft_2023_cybersecurity_risks",
                "query": "What cybersecurity risks does Microsoft describe?",
                "scope": {"sections": ["Item 1A"]},
                "reference_answer": "Cyberthreats are increasingly difficult to detect.",
                "required_facts": [
                    {
                        "fact_id": "cyber_threat_evolution",
                        "claim": "Cyberthreats are increasingly difficult to detect.",
                        "evidence_units": [
                            {
                                "evidence_id": "cyber_threat_evolution_quote",
                                "section": "Item 1A",
                                "quote": EVIDENCE_QUOTE,
                            }
                        ],
                    }
                ],
                "optional_facts": (
                    [
                        {
                            "fact_id": "cyber_customer_limitations",
                            "claim": (
                                "Customer limitations may increase vulnerability."
                            ),
                            "evidence_units": [
                                {
                                    "evidence_id": "cyber_customer_limitations_quote",
                                    "section": "Item 1A",
                                    "quote": OPTIONAL_EVIDENCE_QUOTE,
                                }
                            ],
                        }
                    ]
                    if include_optional_fact
                    else []
                ),
                "annotation": {
                    "method": "manual_source_review",
                    "annotator": "test_author",
                    "review_status": review_status,
                },
            }
        ],
    }
    gold_bytes = json.dumps(gold, ensure_ascii=False, indent=2).encode("utf-8")

    return {
        "gold_bytes": gold_bytes,
        "raw_filing_bytes": raw_bytes,
        "chunks_bytes": chunks_bytes,
        "gold_path": "data/evals/gold/msft_2023_questions.json",
        "raw_filing_path": (
            "data/sec_filings/raw/MSFT_2023_000095017023035122.html"
        ),
        "chunks_path": "data/index/MSFT_2023_chunks.json",
        "index_name": "MSFT_2023",
        "resolution_name": "msft_2023_sentence",
    }


def test_resolver_computes_hashes_offsets_and_chunk_mappings() -> None:
    inputs = build_resolver_inputs()
    artifact = resolve_eval_artifact(**inputs)

    evidence = artifact.questions[0].required_fact_resolutions[
        0
    ].evidence_resolutions[0]
    assert artifact.gold_source.sha256 == sha256_bytes(inputs["gold_bytes"])
    assert evidence.document_char_start == (
        artifact.filing_source.sections[0].document_char_start
        + evidence.section_char_start
    )
    assert evidence.document_char_end - evidence.document_char_start == len(
        EVIDENCE_QUOTE
    )
    assert evidence.matched_chunk_ids
    assert (
        artifact.questions[0].required_evidence_chunk_ids
        == evidence.matched_chunk_ids
    )
    assert (
        artifact.questions[0].relevant_context_chunk_ids
        == evidence.matched_chunk_ids
    )
    assert artifact.validation_summary.status == "passed"


def test_resolver_rejects_non_unique_quote() -> None:
    inputs = build_resolver_inputs(quote_occurrences=2)

    with pytest.raises(ResolutionError, match="found 2 matches"):
        resolve_eval_artifact(**inputs)


def test_resolver_separates_required_and_optional_chunk_unions() -> None:
    inputs = build_resolver_inputs(include_optional_fact=True)
    artifact = resolve_eval_artifact(**inputs)

    question = artifact.questions[0]
    optional_resolution = question.optional_fact_resolutions[0]
    assert question.validation.required_fact_count == 1
    assert question.validation.optional_fact_count == 1
    assert optional_resolution.fact_id == "cyber_customer_limitations"
    assert set(question.required_evidence_chunk_ids).issubset(
        question.relevant_context_chunk_ids
    )
    assert set(optional_resolution.resolved_chunk_ids).issubset(
        question.relevant_context_chunk_ids
    )


def test_resolver_rejects_unverified_question() -> None:
    inputs = build_resolver_inputs(review_status="draft")

    with pytest.raises(ResolutionError, match="must be verified"):
        resolve_eval_artifact(**inputs)


def test_resolver_rejects_when_no_chunk_contains_quote() -> None:
    inputs = build_resolver_inputs()
    chunks = json.loads(inputs["chunks_bytes"])
    first_chunk = chunks[0]
    first_chunk["text"] = "Item 1A"
    first_chunk["char_end"] = first_chunk["char_start"] + len(first_chunk["text"])
    first_chunk["token_count"] = 2
    inputs["chunks_bytes"] = json.dumps(chunks[:1], indent=2).encode("utf-8")

    with pytest.raises(ResolutionError, match="No chunk fully contains evidence"):
        resolve_eval_artifact(**inputs)
