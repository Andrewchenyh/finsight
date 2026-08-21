import pytest

from backend.evals.legacy_retrieval_metrics import evaluate_retrieval_results
from backend.evals.legacy_schemas import EvalQuestionResult


def build_question_result(
    question_id: str,
    expected_chunk_hit: bool,
    expected_chunk_rank: int | None,
    heuristic_precision_at_k: float,
) -> EvalQuestionResult:
    return EvalQuestionResult(
        question_id=question_id,
        retrieval_mode="bm25",
        retrieved_chunk_ids=[],
        expected_chunk_hit=expected_chunk_hit,
        expected_chunk_rank=expected_chunk_rank,
        expected_terms_hit_count=0,
        expected_terms_total=0,
        heuristic_precision_at_k=heuristic_precision_at_k,
    )


def test_aggregate_metrics_use_honest_hit_and_rank_semantics() -> None:
    results = [
        build_question_result("hit", True, 2, 0.6),
        build_question_result("miss", False, None, 0.2),
    ]

    run_result = evaluate_retrieval_results(
        dataset_name="test_dataset",
        retrieval_mode="bm25",
        question_results=results,
    )

    assert run_result.hit_rate_at_k == pytest.approx(0.5)
    assert run_result.mean_heuristic_precision_at_k == pytest.approx(0.4)
    assert run_result.mean_first_hit_rank == pytest.approx(2.0)


def test_mean_first_hit_rank_is_none_when_every_question_misses() -> None:
    run_result = evaluate_retrieval_results(
        dataset_name="test_dataset",
        retrieval_mode="bm25",
        question_results=[build_question_result("miss", False, None, 0.0)],
    )

    assert run_result.hit_rate_at_k == 0.0
    assert run_result.mean_first_hit_rank is None
