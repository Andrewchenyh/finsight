from backend.evals.schemas import (
    QuestionResolution,
    RetrievalQuestionMetrics,
    RetrievalRunMetrics,
)
from backend.schemas import RetrievalMode


def score_retrieval_question(
    *,
    question: QuestionResolution,
    retrieved_chunk_ids: list[str],
    top_k: int,
) -> RetrievalQuestionMetrics:
    """Score one ranked retrieval result against generated fact mappings."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(retrieved_chunk_ids) != len(set(retrieved_chunk_ids)):
        raise ValueError("retrieved_chunk_ids must not contain duplicates.")

    top_chunk_ids = retrieved_chunk_ids[:top_k]
    required_evidence_chunk_ids = question.required_evidence_chunk_ids
    required_evidence_chunk_id_set = set(required_evidence_chunk_ids)
    relevant_context_chunk_ids = question.relevant_context_chunk_ids
    relevant_context_chunk_id_set = set(relevant_context_chunk_ids)
    relevant_context_retrieved_chunk_ids = [
        chunk_id
        for chunk_id in top_chunk_ids
        if chunk_id in relevant_context_chunk_id_set
    ]

    retrieved_chunk_id_set = set(top_chunk_ids)
    covered_fact_ids = [
        fact.fact_id
        for fact in question.required_fact_resolutions
        if retrieved_chunk_id_set.intersection(fact.resolved_chunk_ids)
    ]
    required_fact_count = len(question.required_fact_resolutions)
    covered_fact_count = len(covered_fact_ids)

    first_required_evidence_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(top_chunk_ids, start=1)
            if chunk_id in required_evidence_chunk_id_set
        ),
        None,
    )

    return RetrievalQuestionMetrics(
        question_id=question.question_id,
        top_k=top_k,
        retrieved_chunk_ids=top_chunk_ids,
        required_evidence_chunk_ids=required_evidence_chunk_ids,
        relevant_context_chunk_ids=relevant_context_chunk_ids,
        relevant_context_retrieved_chunk_ids=(
            relevant_context_retrieved_chunk_ids
        ),
        covered_fact_ids=covered_fact_ids,
        required_fact_count=required_fact_count,
        covered_fact_count=covered_fact_count,
        required_fact_hit_at_k=covered_fact_count > 0,
        fact_recall_at_k=covered_fact_count / required_fact_count,
        full_coverage_at_k=covered_fact_count == required_fact_count,
        context_precision_at_k=(
            len(relevant_context_retrieved_chunk_ids) / top_k
        ),
        first_required_evidence_rank=first_required_evidence_rank,
        required_evidence_reciprocal_rank_at_k=(
            1.0 / first_required_evidence_rank
            if first_required_evidence_rank is not None
            else 0.0
        ),
    )


def aggregate_retrieval_metrics(
    *,
    resolution_name: str,
    retrieval_mode: RetrievalMode,
    top_k: int,
    question_results: list[RetrievalQuestionMetrics],
) -> RetrievalRunMetrics:
    """Macro-average question metrics for one retrieval mode and cutoff."""
    if not question_results:
        raise ValueError("question_results must not be empty.")

    question_count = len(question_results)
    return RetrievalRunMetrics(
        resolution_name=resolution_name,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        question_results=question_results,
        required_fact_hit_rate_at_k=sum(
            result.required_fact_hit_at_k for result in question_results
        )
        / question_count,
        macro_fact_recall_at_k=sum(
            result.fact_recall_at_k for result in question_results
        )
        / question_count,
        full_coverage_rate_at_k=sum(
            result.full_coverage_at_k for result in question_results
        )
        / question_count,
        mean_context_precision_at_k=sum(
            result.context_precision_at_k for result in question_results
        )
        / question_count,
        required_evidence_mrr_at_k=sum(
            result.required_evidence_reciprocal_rank_at_k
            for result in question_results
        )
        / question_count,
    )
