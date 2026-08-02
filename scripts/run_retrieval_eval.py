import argparse
import json
from pathlib import Path

from backend.evals.retrieval_metrics import (
    evaluate_retrieval_results,
    score_question_result,
)
from backend.evals.schemas import EvalDataset
from backend.service import retrieve_sec_chunks
from backend.schemas import RetrievalMode


DEFAULT_MODES: list[RetrievalMode] = [
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
]


def run_eval(dataset_path: Path, modes: list[RetrievalMode]) -> None:
    dataset = EvalDataset.model_validate(
        json.loads(dataset_path.read_text(encoding="utf-8"))
    )

    print(f"Dataset: {dataset.name}")
    print(f"Index: {dataset.index_name}")
    print(f"Questions: {len(dataset.questions)}")
    print()

    for mode in modes:
        question_results = []

        for question in dataset.questions:
            retrieved_chunks = retrieve_sec_chunks(
                query=question.query,
                index_name=dataset.index_name,
                ticker=question.ticker,
                fiscal_year=question.fiscal_year,
                section=question.section,
                top_k=question.top_k,
                retrieval_mode=mode,
            )

            result = score_question_result(
                question=question,
                retrieved_chunks=retrieved_chunks,
                retrieval_mode=mode,
            )
            question_results.append(result)

        run_result = evaluate_retrieval_results(
            dataset_name=dataset.name,
            retrieval_mode=mode,
            question_results=question_results,
        )

        print("=" * 100)
        print(f"Mode: {mode}")
        print(f"Recall@k: {run_result.recall_at_k:.2f}")
        print(f"Mean precision@k: {run_result.mean_precision_at_k:.2f}")

        if run_result.mean_expected_chunk_rank is not None:
            print(f"Mean expected chunk rank: {run_result.mean_expected_chunk_rank:.2f}")
        else:
            print("Mean expected chunk rank: n/a")

        print()
        for question_result in run_result.question_results:
            rank = question_result.expected_chunk_rank or "miss"
            print(
                f"{question_result.question_id}: "
                f"rank={rank}, "
                f"terms={question_result.expected_terms_hit_count}/"
                f"{question_result.expected_terms_total}, "
                f"precision={question_result.precision_at_k:.2f}"
            )
        print()


def parse_modes(raw_modes: list[str]) -> list[RetrievalMode]:
    valid_modes = set(DEFAULT_MODES)
    modes: list[RetrievalMode] = []

    for raw_mode in raw_modes:
        if raw_mode not in valid_modes:
            raise ValueError(f"Unsupported retrieval mode: {raw_mode}")
        modes.append(raw_mode)  # type: ignore[arg-type]

    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinSight retrieval evaluation.")
    parser.add_argument(
        "--dataset",
        default="data/evals/msft_2023_retrieval.json",
        help="Path to eval dataset JSON.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        help="Retrieval modes to evaluate.",
    )

    args = parser.parse_args()
    modes = parse_modes(args.modes)

    run_eval(dataset_path=Path(args.dataset), modes=modes)


if __name__ == "__main__":
    main()