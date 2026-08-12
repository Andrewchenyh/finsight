import argparse
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from backend.evals.resolver import ResolutionError, resolve_eval_artifact
from backend.evals.schemas import ResolvedEvalArtifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = "data/evals/gold/msft_2023_questions.json"
DEFAULT_RAW_FILING_PATH = (
    "data/sec_filings/raw/MSFT_2023_000095017023035122.html"
)
DEFAULT_CHUNKS_PATH = "data/index/MSFT_2023_chunks.json"
DEFAULT_OUTPUT_PATH = "data/evals/resolved/msft_2023_sentence.json"


def resolve_and_write(
    *,
    gold_path: str,
    raw_filing_path: str,
    chunks_path: str,
    output_path: str,
    index_name: str,
    resolution_name: str,
) -> ResolvedEvalArtifact:
    """Read resolver inputs, validate them, and atomically write the artifact."""
    gold_file, gold_repo_path = _resolve_repository_path(gold_path)
    raw_filing_file, raw_filing_repo_path = _resolve_repository_path(
        raw_filing_path
    )
    chunks_file, chunks_repo_path = _resolve_repository_path(chunks_path)
    output_file, _ = _resolve_repository_path(output_path)

    artifact = resolve_eval_artifact(
        gold_bytes=gold_file.read_bytes(),
        raw_filing_bytes=raw_filing_file.read_bytes(),
        chunks_bytes=chunks_file.read_bytes(),
        gold_path=gold_repo_path,
        raw_filing_path=raw_filing_repo_path,
        chunks_path=chunks_repo_path,
        index_name=index_name,
        resolution_name=resolution_name,
    )

    serialized = (artifact.model_dump_json(indent=2) + "\n").encode("utf-8")
    _write_bytes_atomically(output_file, serialized)
    return artifact


def _resolve_repository_path(raw_path: str) -> tuple[Path, str]:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the repository: {raw_path}") from exc

    return resolved, relative.as_posix()


def _write_bytes_atomically(output_path: Path, content: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.replace(output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve source-grounded eval evidence to one exact chunk index."
    )
    parser.add_argument("--gold", default=DEFAULT_GOLD_PATH)
    parser.add_argument("--filing", default=DEFAULT_RAW_FILING_PATH)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--index-name", default="MSFT_2023")
    parser.add_argument("--resolution-name", default="msft_2023_sentence")
    args = parser.parse_args()

    try:
        artifact = resolve_and_write(
            gold_path=args.gold,
            raw_filing_path=args.filing,
            chunks_path=args.chunks,
            output_path=args.output,
            index_name=args.index_name,
            resolution_name=args.resolution_name,
        )
    except (OSError, ResolutionError, ValidationError, ValueError) as exc:
        parser.exit(status=1, message=f"Resolution failed: {exc}\n")

    print(f"Resolved artifact: {args.output}")
    print(f"Questions: {artifact.validation_summary.question_count}")
    print(f"Required facts: {artifact.validation_summary.required_fact_count}")
    print(f"Optional facts: {artifact.validation_summary.optional_fact_count}")
    print(f"Evidence units: {artifact.validation_summary.evidence_count}")
    print(f"Chunks: {artifact.index_source.chunk_count}")
    print(f"Validation: {artifact.validation_summary.status}")


if __name__ == "__main__":
    main()
