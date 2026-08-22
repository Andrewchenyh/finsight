from pathlib import Path

import pytest

from scripts.resolve_retrieval_gold import (
    REPOSITORY_ROOT,
    _resolve_repository_path,
    _write_bytes_atomically,
)


def test_atomic_writer_replaces_complete_output(tmp_path: Path) -> None:
    output_path = tmp_path / "resolved.json"
    output_path.write_bytes(b"old-content")

    _write_bytes_atomically(output_path, b"new-complete-content")

    assert output_path.read_bytes() == b"new-complete-content"
    assert list(tmp_path.glob("*.tmp")) == []


def test_repository_path_is_stored_as_portable_relative_path() -> None:
    resolved, relative = _resolve_repository_path(
        "data/evals/gold/msft_2023_questions.json"
    )

    assert resolved == (
        REPOSITORY_ROOT / "data/evals/gold/msft_2023_questions.json"
    ).resolve()
    assert relative == "data/evals/gold/msft_2023_questions.json"


def test_repository_path_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        _resolve_repository_path("../outside.json")
