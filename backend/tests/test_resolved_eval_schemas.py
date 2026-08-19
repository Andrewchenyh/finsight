from copy import deepcopy

import pytest
from pydantic import ValidationError

from backend.evals.schemas import ResolvedEvalArtifact


def build_resolved_artifact_dict() -> dict:
    required_chunk_ids = [
        "MSFT_2023_Item1A_10_d4daa62ff465",
        "MSFT_2023_Item1A_11_3859158f318b",
    ]
    optional_chunk_ids = ["MSFT_2023_Item1A_13_64f13a7774a9"]
    return {
        "schema_version": "1.0",
        "artifact_type": "resolved_retrieval_gold",
        "resolution_name": "msft_2023_sentence",
        "resolver": {
            "name": "exact_quote_chunk_resolver",
            "version": "1.0",
            "quote_match_policy": "unique_exact_match",
            "chunk_match_policy": "full_quote_containment",
        },
        "gold_source": {
            "dataset_name": "msft_2023_retrieval",
            "path": "data/evals/gold/msft_2023_questions.json",
            "sha256": "a" * 64,
        },
        "filing_source": {
            "ticker": "MSFT",
            "fiscal_year": 2023,
            "filing_type": "10-K",
            "accession_number": "0000950170-23-035122",
            "raw_path": "data/sec_filings/raw/MSFT_2023_000095017023035122.html",
            "raw_sha256": "b" * 64,
            "normalization_method": "clean_filing_html_v1",
            "sections": [
                {
                    "section": "Item 1A",
                    "document_char_start": 116652,
                    "document_char_end": 160000,
                    "normalized_text_sha256": "c" * 64,
                }
            ],
        },
        "index_source": {
            "index_name": "MSFT_2023",
            "chunks_path": "data/index/MSFT_2023_chunks.json",
            "chunks_sha256": "d" * 64,
            "chunk_count": 193,
        },
        "questions": [
            {
                "question_id": "msft_2023_cybersecurity_risks",
                "required_fact_resolutions": [
                    {
                        "fact_id": "cyber_threat_evolution",
                        "evidence_resolutions": [
                            {
                                "evidence_id": "cyber_threat_evolution_quote",
                                "section": "Item 1A",
                                "quote_sha256": (
                                    "4870cb5e4a7257240b0ffae12d64439e603393ed215c95"
                                    "afca8cf610097097ab"
                                ),
                                "section_char_start": 17478,
                                "section_char_end": 17647,
                                "document_char_start": 134130,
                                "document_char_end": 134299,
                                "exact_match_count": 1,
                                "match_method": "exact",
                                "matched_chunk_ids": required_chunk_ids,
                            }
                        ],
                        "resolved_chunk_ids": required_chunk_ids,
                    }
                ],
                "optional_fact_resolutions": [
                    {
                        "fact_id": "cyber_optional_context",
                        "evidence_resolutions": [
                            {
                                "evidence_id": "cyber_optional_context_quote",
                                "section": "Item 1A",
                                "quote_sha256": "e" * 64,
                                "section_char_start": 18000,
                                "section_char_end": 18010,
                                "document_char_start": 134652,
                                "document_char_end": 134662,
                                "exact_match_count": 1,
                                "match_method": "exact",
                                "matched_chunk_ids": optional_chunk_ids,
                            }
                        ],
                        "resolved_chunk_ids": optional_chunk_ids,
                    }
                ],
                "required_evidence_chunk_ids": required_chunk_ids,
                "relevant_context_chunk_ids": (
                    required_chunk_ids + optional_chunk_ids
                ),
                "validation": {
                    "required_fact_count": 1,
                    "resolved_required_fact_count": 1,
                    "optional_fact_count": 1,
                    "resolved_optional_fact_count": 1,
                    "evidence_count": 2,
                    "resolved_evidence_count": 2,
                    "status": "passed",
                },
            }
        ],
        "validation_summary": {
            "question_count": 1,
            "required_fact_count": 1,
            "optional_fact_count": 1,
            "evidence_count": 2,
            "unresolved_evidence_ids": [],
            "status": "passed",
        },
    }


def test_valid_resolved_artifact_contract() -> None:
    artifact = ResolvedEvalArtifact.model_validate(build_resolved_artifact_dict())

    resolution = artifact.questions[0].required_fact_resolutions[0]
    assert resolution.resolved_chunk_ids == resolution.evidence_resolutions[0].matched_chunk_ids


def test_resolved_artifact_rejects_incorrect_question_chunk_union() -> None:
    raw_artifact = build_resolved_artifact_dict()
    raw_artifact["questions"][0]["required_evidence_chunk_ids"] = [
        "incorrect_chunk"
    ]

    with pytest.raises(ValidationError, match="ordered required-fact chunk union"):
        ResolvedEvalArtifact.model_validate(raw_artifact)


def test_resolved_artifact_rejects_incorrect_context_chunk_union() -> None:
    raw_artifact = build_resolved_artifact_dict()
    raw_artifact["questions"][0]["relevant_context_chunk_ids"] = [
        "incorrect_chunk"
    ]

    with pytest.raises(ValidationError, match="ordered all-fact chunk union"):
        ResolvedEvalArtifact.model_validate(raw_artifact)


def test_resolved_artifact_rejects_inconsistent_document_offsets() -> None:
    raw_artifact = build_resolved_artifact_dict()
    evidence = raw_artifact["questions"][0]["required_fact_resolutions"][0][
        "evidence_resolutions"
    ][0]
    evidence["document_char_start"] += 1
    evidence["document_char_end"] += 1

    with pytest.raises(ValidationError, match="do not match section offsets"):
        ResolvedEvalArtifact.model_validate(raw_artifact)


def test_passed_resolved_artifact_rejects_unresolved_evidence() -> None:
    raw_artifact = deepcopy(build_resolved_artifact_dict())
    raw_artifact["validation_summary"]["unresolved_evidence_ids"] = [
        "unresolved_quote"
    ]

    with pytest.raises(ValidationError, match="cannot contain unresolved evidence"):
        ResolvedEvalArtifact.model_validate(raw_artifact)
