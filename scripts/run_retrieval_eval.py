import argparse
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from backend.evals.resolver import sha256_bytes, sha256_text
from backend.evals.retrieval_metrics import (
    aggregate_retrieval_metrics,
    score_retrieval_question,
)
from backend.evals.schemas import (
    GoldEvalDataset,
    ResolvedEvalArtifact,
    RetrievalRunMetrics,
)
from backend.schemas import RetrievedChunk, RetrievalMode
from backend.service import retrieve_sec_chunks


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = "data/evals/gold/msft_2023_questions.json"
DEFAULT_RESOLVED_PATH = "data/evals/resolved/msft_2023_sentence.json"
DEFAULT_MODES: list[RetrievalMode] = [
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
]
DEFAULT_CUTOFFS = [1, 3, 5]

RetrieverFunction = Callable[..., list[RetrievedChunk]]


def load_eval_inputs(
    *,
    gold_path: Path,
    resolved_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[GoldEvalDataset, ResolvedEvalArtifact]:
    """Load and cross-check gold, resolved, and current chunk-index inputs."""
    gold_bytes = gold_path.read_bytes()
    resolved_bytes = resolved_path.read_bytes()
    gold_dataset = GoldEvalDataset.model_validate_json(gold_bytes)
    resolved_artifact = ResolvedEvalArtifact.model_validate_json(resolved_bytes)

    _validate_gold_resolution_compatibility(
        gold_dataset=gold_dataset,
        gold_bytes=gold_bytes,
        resolved_artifact=resolved_artifact,
    )

    chunks_path = _resolve_artifact_path(
        resolved_artifact.index_source.chunks_path,
        repository_root=repository_root,
    )
    expected_service_chunks_path = (
        repository_root
        / "data"
        / "index"
        / f"{resolved_artifact.index_source.index_name}_chunks.json"
    ).resolve()
    if chunks_path != expected_service_chunks_path:
        raise ValueError(
            "Resolved chunks path is not the path used by the retrieval service."
        )

    chunks_bytes = chunks_path.read_bytes()
    if sha256_bytes(chunks_bytes) != resolved_artifact.index_source.chunks_sha256:
        raise ValueError(
            "Chunk index hash does not match the resolved artifact; rerun the resolver."
        )

    chunks_payload = json.loads(chunks_bytes)
    if not isinstance(chunks_payload, list):
        raise ValueError("Chunk index must be a JSON list.")
    if len(chunks_payload) != resolved_artifact.index_source.chunk_count:
        raise ValueError("Chunk count does not match the resolved artifact.")

    current_chunk_ids = {
        chunk.get("chunk_id")
        for chunk in chunks_payload
        if isinstance(chunk, dict)
    }
    resolved_chunk_ids = {
        chunk_id
        for question in resolved_artifact.questions
        for chunk_id in question.relevant_context_chunk_ids
    }
    missing_chunk_ids = sorted(resolved_chunk_ids - current_chunk_ids)
    if missing_chunk_ids:
        raise ValueError(
            "Resolved expected chunks are absent from the current index: "
            + ", ".join(missing_chunk_ids)
        )

    return gold_dataset, resolved_artifact


def evaluate_retrieval_modes(
    *,
    gold_dataset: GoldEvalDataset,
    resolved_artifact: ResolvedEvalArtifact,
    modes: list[RetrievalMode],
    cutoffs: list[int],
    retrieve: RetrieverFunction = retrieve_sec_chunks,
) -> list[RetrievalRunMetrics]:
    """Retrieve once at max(k), then score every requested cutoff."""
    normalized_cutoffs = normalize_cutoffs(cutoffs)
    maximum_k = max(normalized_cutoffs)
    gold_questions = {question.id: question for question in gold_dataset.questions}
    run_results: list[RetrievalRunMetrics] = []

    for mode in modes:
        results_by_cutoff = {cutoff: [] for cutoff in normalized_cutoffs}

        for question_resolution in resolved_artifact.questions:
            gold_question = gold_questions[question_resolution.question_id]
            if len(gold_question.scope.sections) != 1:
                raise ValueError(
                    "The current retrieval service supports exactly one section "
                    f"per eval question: {gold_question.id}"
                )

            retrieved_chunks = retrieve(
                query=gold_question.query,
                index_name=resolved_artifact.index_source.index_name,
                ticker=gold_dataset.filing.ticker,
                fiscal_year=gold_dataset.filing.fiscal_year,
                section=gold_question.scope.sections[0],
                filing_type=gold_dataset.filing.filing_type,
                top_k=maximum_k,
                retrieval_mode=mode,
            )
            retrieved_chunk_ids = [result.chunk.chunk_id for result in retrieved_chunks]

            for cutoff in normalized_cutoffs:
                results_by_cutoff[cutoff].append(
                    score_retrieval_question(
                        question=question_resolution,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        top_k=cutoff,
                    )
                )

        for cutoff in normalized_cutoffs:
            run_results.append(
                aggregate_retrieval_metrics(
                    resolution_name=resolved_artifact.resolution_name,
                    retrieval_mode=mode,
                    top_k=cutoff,
                    question_results=results_by_cutoff[cutoff],
                )
            )

    return run_results


def normalize_cutoffs(raw_cutoffs: list[int]) -> list[int]:
    if not raw_cutoffs:
        raise ValueError("At least one cutoff is required.")
    if any(cutoff <= 0 for cutoff in raw_cutoffs):
        raise ValueError("Cutoffs must be positive integers.")
    if len(raw_cutoffs) != len(set(raw_cutoffs)):
        raise ValueError("Cutoffs must not contain duplicates.")
    return sorted(raw_cutoffs)


def parse_modes(raw_modes: list[str]) -> list[RetrievalMode]:
    if not raw_modes:
        raise ValueError("At least one retrieval mode is required.")
    valid_modes = set(DEFAULT_MODES)
    if len(raw_modes) != len(set(raw_modes)):
        raise ValueError("Retrieval modes must not contain duplicates.")

    unsupported_modes = [mode for mode in raw_modes if mode not in valid_modes]
    if unsupported_modes:
        raise ValueError(
            "Unsupported retrieval modes: " + ", ".join(unsupported_modes)
        )
    return raw_modes  # type: ignore[return-value]


def print_results(
    *,
    gold_dataset: GoldEvalDataset,
    resolved_artifact: ResolvedEvalArtifact,
    run_results: list[RetrievalRunMetrics],
) -> None:
    print(f"Dataset: {gold_dataset.dataset_name}")
    print(f"Resolution: {resolved_artifact.resolution_name}")
    print(f"Index: {resolved_artifact.index_source.index_name}")
    print(f"Questions: {len(resolved_artifact.questions)}")

    modes = list(dict.fromkeys(result.retrieval_mode for result in run_results))
    for mode in modes:
        mode_results = [
            result for result in run_results if result.retrieval_mode == mode
        ]
        print()
        print("=" * 100)
        print(f"Mode: {mode}")
        print(
            "k  Required Hit  Fact Recall  Full Coverage  Context Precision  Required MRR"
        )
        for result in mode_results:
            print(
                f"{result.top_k:<2} "
                f"{result.required_fact_hit_rate_at_k:>12.2f}  "
                f"{result.macro_fact_recall_at_k:>11.2f}  "
                f"{result.full_coverage_rate_at_k:>13.2f}  "
                f"{result.mean_context_precision_at_k:>17.2f}  "
                f"{result.required_evidence_mrr_at_k:>12.2f}"
            )

        maximum_result = max(mode_results, key=lambda result: result.top_k)
        print(f"Per-question details at k={maximum_result.top_k}:")
        for question_result in maximum_result.question_results:
            rank = question_result.first_required_evidence_rank or "miss"
            print(
                f"  {question_result.question_id}: "
                f"facts={question_result.covered_fact_count}/"
                f"{question_result.required_fact_count}, "
                "relevant_context_chunks="
                f"{len(question_result.relevant_context_retrieved_chunk_ids)}, "
                f"first_required_rank={rank}"
            )


def _validate_gold_resolution_compatibility(
    *,
    gold_dataset: GoldEvalDataset,
    gold_bytes: bytes,
    resolved_artifact: ResolvedEvalArtifact,
) -> None:
    if resolved_artifact.gold_source.sha256 != sha256_bytes(gold_bytes):
        raise ValueError(
            "Gold file hash does not match the resolved artifact; rerun the resolver."
        )
    if resolved_artifact.gold_source.dataset_name != gold_dataset.dataset_name:
        raise ValueError("Gold dataset name does not match the resolved artifact.")

    filing_fields = ("ticker", "fiscal_year", "filing_type", "accession_number")
    for field_name in filing_fields:
        if getattr(gold_dataset.filing, field_name) != getattr(
            resolved_artifact.filing_source,
            field_name,
        ):
            raise ValueError(
                f"Gold filing {field_name} does not match the resolved artifact."
            )

    gold_question_ids = [question.id for question in gold_dataset.questions]
    resolved_question_ids = [
        question.question_id for question in resolved_artifact.questions
    ]
    if gold_question_ids != resolved_question_ids:
        raise ValueError("Gold and resolved question IDs or ordering do not match.")

    for gold_question, resolved_question in zip(
        gold_dataset.questions,
        resolved_artifact.questions,
        strict=True,
    ):
        fact_groups = (
            (
                "required",
                gold_question.required_facts,
                resolved_question.required_fact_resolutions,
            ),
            (
                "optional",
                gold_question.optional_facts,
                resolved_question.optional_fact_resolutions,
            ),
        )
        for fact_kind, gold_facts, resolved_facts in fact_groups:
            gold_fact_ids = [fact.fact_id for fact in gold_facts]
            resolved_fact_ids = [fact.fact_id for fact in resolved_facts]
            if gold_fact_ids != resolved_fact_ids:
                raise ValueError(
                    f"{fact_kind.title()} fact IDs do not match for "
                    f"question {gold_question.id!r}."
                )

            for gold_fact, resolved_fact in zip(
                gold_facts,
                resolved_facts,
                strict=True,
            ):
                gold_evidence_ids = [
                    evidence.evidence_id for evidence in gold_fact.evidence_units
                ]
                resolved_evidence_ids = [
                    evidence.evidence_id
                    for evidence in resolved_fact.evidence_resolutions
                ]
                if gold_evidence_ids != resolved_evidence_ids:
                    raise ValueError(
                        f"Evidence IDs do not match for fact {gold_fact.fact_id!r}."
                    )

                for gold_evidence, resolved_evidence in zip(
                    gold_fact.evidence_units,
                    resolved_fact.evidence_resolutions,
                    strict=True,
                ):
                    if gold_evidence.section != resolved_evidence.section:
                        raise ValueError(
                            "Evidence section does not match for "
                            f"{gold_evidence.evidence_id!r}."
                        )
                    if (
                        sha256_text(gold_evidence.quote)
                        != resolved_evidence.quote_sha256
                    ):
                        raise ValueError(
                            "Evidence quote hash does not match for "
                            f"{gold_evidence.evidence_id!r}."
                        )


def _resolve_artifact_path(raw_path: str, *, repository_root: Path) -> Path:
    resolved = (repository_root / raw_path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Resolved artifact path escapes repository: {raw_path}"
        ) from exc
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-grounded FinSight retrieval evaluation."
    )
    parser.add_argument("--gold", default=DEFAULT_GOLD_PATH)
    parser.add_argument("--resolved", default=DEFAULT_RESOLVED_PATH)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=DEFAULT_CUTOFFS)
    args = parser.parse_args()

    try:
        modes = parse_modes(args.modes)
        cutoffs = normalize_cutoffs(args.cutoffs)
        gold_dataset, resolved_artifact = load_eval_inputs(
            gold_path=Path(args.gold),
            resolved_path=Path(args.resolved),
        )
        run_results = evaluate_retrieval_modes(
            gold_dataset=gold_dataset,
            resolved_artifact=resolved_artifact,
            modes=modes,
            cutoffs=cutoffs,
        )
    except (OSError, ValidationError, ValueError) as exc:
        parser.exit(status=1, message=f"Evaluation failed: {exc}\n")

    print_results(
        gold_dataset=gold_dataset,
        resolved_artifact=resolved_artifact,
        run_results=run_results,
    )


if __name__ == "__main__":
    main()
