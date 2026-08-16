from backend.evals.legacy_schemas import EvalQuestion, EvalQuestionResult, EvalRunResult
from backend.schemas import RetrievalMode, RetrievedChunk


def evaluate_retrieval_results(
    dataset_name: str,
    retrieval_mode: RetrievalMode,
    question_results: list[EvalQuestionResult],
) -> EvalRunResult:
    """Aggregate retrieval metrics for one retrieval mode."""
    if not question_results:
        raise ValueError("question_results must not be empty.")

    hit_rate_at_k = sum(result.expected_chunk_hit for result in question_results) / len(question_results)

    mean_heuristic_precision_at_k = sum(
        result.heuristic_precision_at_k for result in question_results
    ) / len(question_results)

    ranks = [
        result.expected_chunk_rank
        for result in question_results
        if result.expected_chunk_rank is not None
    ]
    mean_first_hit_rank = sum(ranks) / len(ranks) if ranks else None

    return EvalRunResult(
        dataset_name=dataset_name,
        retrieval_mode=retrieval_mode,
        question_results=question_results,
        hit_rate_at_k=hit_rate_at_k,
        mean_heuristic_precision_at_k=mean_heuristic_precision_at_k,
        mean_first_hit_rank=mean_first_hit_rank,
    )


def score_question_result(
    question: EvalQuestion,
    retrieved_chunks: list[RetrievedChunk],
    retrieval_mode: RetrievalMode,
) -> EvalQuestionResult:
    """Score one query's retrieved chunks against expected IDs/terms."""
    retrieved_chunk_ids = [result.chunk.chunk_id for result in retrieved_chunks]

    expected_chunk_rank = find_first_expected_chunk_rank(
        expected_chunk_ids=question.expected_chunk_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

    expected_chunk_hit = expected_chunk_rank is not None

    expected_terms_hit_count = count_expected_terms(
        expected_terms=question.expected_terms,
        retrieved_chunks=retrieved_chunks,
    )

    expected_terms_total = len(question.expected_terms)

    heuristic_precision_at_k = estimate_heuristic_precision_at_k(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return EvalQuestionResult(
        question_id=question.id,
        retrieval_mode=retrieval_mode,
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_hit=expected_chunk_hit,
        expected_chunk_rank=expected_chunk_rank,
        expected_terms_hit_count=expected_terms_hit_count,
        expected_terms_total=expected_terms_total,
        heuristic_precision_at_k=heuristic_precision_at_k,
    )


def find_first_expected_chunk_rank(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> int | None:
    """Return one-based rank of the first expected chunk hit."""
    if not expected_chunk_ids:
        return None

    expected = set(expected_chunk_ids)

    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return index

    return None


def count_expected_terms(
    expected_terms: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> int:
    """Count expected terms that appear anywhere in retrieved chunk text."""
    if not expected_terms:
        return 0

    combined_text = "\n".join(result.chunk.text for result in retrieved_chunks).lower()

    return sum(1 for term in expected_terms if term.lower() in combined_text)


def estimate_heuristic_precision_at_k(
    question: EvalQuestion,
    retrieved_chunks: list[RetrievedChunk],
) -> float:
    """Estimate heuristic precision@k from expected IDs and term matches.

    A chunk is counted relevant if:
    - its chunk ID is listed as expected, or
    - it contains at least one expected term.
    """
    if not retrieved_chunks:
        return 0.0

    expected_chunk_ids = set(question.expected_chunk_ids)
    expected_terms = [term.lower() for term in question.expected_terms]

    relevant_count = 0

    for result in retrieved_chunks:
        chunk_id_hit = result.chunk.chunk_id in expected_chunk_ids
        chunk_text = result.chunk.text.lower()
        term_hit = any(term in chunk_text for term in expected_terms)

        if chunk_id_hit or term_hit:
            relevant_count += 1

    return relevant_count / len(retrieved_chunks)
