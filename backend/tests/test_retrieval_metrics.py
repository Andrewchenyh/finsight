from pathlib import Path

import pytest

from backend.evals.retrieval_metrics import (
    aggregate_retrieval_metrics,
    score_retrieval_question,
)
from backend.evals.schemas import ResolvedEvalArtifact


RESOLVED_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evals"
    / "resolved"
    / "msft_2023_sentence.json"
)


def load_cybersecurity_resolution():
    artifact = ResolvedEvalArtifact.model_validate_json(
        RESOLVED_DATASET_PATH.read_text(encoding="utf-8")
    )
    return artifact, artifact.questions[0]


def test_one_chunk_can_cover_multiple_required_facts() -> None:
    _, question = load_cybersecurity_resolution()

    result = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=["MSFT_2023_Item1A_11_3859158f318b"],
        top_k=5,
    )

    assert result.covered_fact_ids == [
        "cyber_threat_evolution",
        "cyber_breach_consequences",
        "cyber_supply_chain_attacks",
    ]
    assert result.fact_recall_at_k == pytest.approx(0.75)
    assert result.context_precision_at_k == pytest.approx(0.20)
    assert result.required_fact_hit_at_k is True
    assert result.full_coverage_at_k is False
    assert result.first_required_evidence_rank == 1
    assert result.required_evidence_reciprocal_rank_at_k == pytest.approx(1.0)


def test_all_required_chunks_produce_full_fact_coverage() -> None:
    _, question = load_cybersecurity_resolution()

    result = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=question.required_evidence_chunk_ids,
        top_k=5,
    )

    assert result.covered_fact_count == 4
    assert result.fact_recall_at_k == pytest.approx(1.0)
    assert result.full_coverage_at_k is True
    assert result.context_precision_at_k == pytest.approx(0.60)


def test_retrieval_miss_scores_zero() -> None:
    _, question = load_cybersecurity_resolution()

    result = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=["unrelated_chunk"],
        top_k=5,
    )

    assert result.required_fact_hit_at_k is False
    assert result.fact_recall_at_k == pytest.approx(0.0)
    assert result.context_precision_at_k == pytest.approx(0.0)
    assert result.first_required_evidence_rank is None
    assert result.required_evidence_reciprocal_rank_at_k == pytest.approx(0.0)


def test_optional_context_improves_precision_without_fact_coverage() -> None:
    _, question = load_cybersecurity_resolution()

    result = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=["MSFT_2023_Item1A_13_64f13a7774a9"],
        top_k=5,
    )

    assert result.required_fact_hit_at_k is False
    assert result.fact_recall_at_k == pytest.approx(0.0)
    assert result.full_coverage_at_k is False
    assert result.context_precision_at_k == pytest.approx(0.20)
    assert result.first_required_evidence_rank is None
    assert result.required_evidence_reciprocal_rank_at_k == pytest.approx(0.0)


def test_aggregate_metrics_macro_average_question_results() -> None:
    artifact, question = load_cybersecurity_resolution()
    full = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=question.required_evidence_chunk_ids,
        top_k=5,
    ).model_copy(update={"question_id": "full"})
    partial = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=["MSFT_2023_Item1A_11_3859158f318b"],
        top_k=5,
    ).model_copy(update={"question_id": "partial"})
    miss = score_retrieval_question(
        question=question,
        retrieved_chunk_ids=["unrelated_chunk"],
        top_k=5,
    ).model_copy(update={"question_id": "miss"})

    result = aggregate_retrieval_metrics(
        resolution_name=artifact.resolution_name,
        retrieval_mode="bm25",
        top_k=5,
        question_results=[full, partial, miss],
    )

    assert result.required_fact_hit_rate_at_k == pytest.approx(2 / 3)
    assert result.macro_fact_recall_at_k == pytest.approx(7 / 12)
    assert result.full_coverage_rate_at_k == pytest.approx(1 / 3)
    assert result.mean_context_precision_at_k == pytest.approx(4 / 15)
    assert result.required_evidence_mrr_at_k == pytest.approx(2 / 3)


def test_duplicate_retrieved_chunk_ids_are_rejected() -> None:
    _, question = load_cybersecurity_resolution()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        score_retrieval_question(
            question=question,
            retrieved_chunk_ids=["duplicate", "duplicate"],
            top_k=5,
        )
