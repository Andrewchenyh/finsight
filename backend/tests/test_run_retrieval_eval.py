import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.evals.resolver import sha256_bytes
from backend.evals.schemas import GoldEvalDataset, ResolvedEvalArtifact
from scripts.run_retrieval_eval import (
    evaluate_retrieval_modes,
    load_eval_inputs,
    normalize_cutoffs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = REPOSITORY_ROOT / "data/evals/gold/msft_2023_questions.json"
RESOLVED_PATH = REPOSITORY_ROOT / "data/evals/resolved/msft_2023_sentence.json"


def load_models() -> tuple[GoldEvalDataset, ResolvedEvalArtifact]:
    gold = GoldEvalDataset.model_validate_json(GOLD_PATH.read_bytes())
    resolved = ResolvedEvalArtifact.model_validate_json(RESOLVED_PATH.read_bytes())
    return gold, resolved


def test_runner_retrieves_once_at_largest_cutoff() -> None:
    gold, resolved = load_models()
    calls: list[dict] = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs)
        chunk_ids_by_query_term = {
            "cybersecurity": "MSFT_2023_Item1A_11_3859158f318b",
            "AI": "MSFT_2023_Item1A_21_4b998f5dff7d",
            "competition": "MSFT_2023_Item1_26_ecaa3ce71e10",
            "cloud infrastructure": "MSFT_2023_Item1A_22_49bb423e539e",
            "economic conditions": "MSFT_2023_Item7_2_85a77caa6ac9",
            "foreign-exchange": "MSFT_2023_Item7_3_1c7eb6621222",
            "revenue growth": "MSFT_2023_Item7_1_517e538f81f7",
            "recognize revenue": "MSFT_2023_Item8_6_3a244e509d86",
            "market risks": "MSFT_2023_Item7A_0_5c716780be3f",
            "liquidity position": "MSFT_2023_Item7_20_e89d57ed98fe",
        }
        chunk_id = next(
            chunk_id
            for query_term, chunk_id in chunk_ids_by_query_term.items()
            if query_term in kwargs["query"]
        )
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(chunk_id=chunk_id)
            )
        ]

    results = evaluate_retrieval_modes(
        gold_dataset=gold,
        resolved_artifact=resolved,
        modes=["bm25"],
        cutoffs=[5, 1, 3],
        retrieve=fake_retrieve,
    )

    assert len(calls) == len(resolved.questions)
    assert all(call["top_k"] == 5 for call in calls)
    expected_sections = {
        question.query: question.scope.sections[0]
        for question in gold.questions
    }
    assert all(
        call["section"] == expected_sections[call["query"]]
        for call in calls
    )
    assert [result.top_k for result in results] == [1, 3, 5]
    expected_macro_recall = (
        3 / 4
        + 3 / 4
        + 2 / 7
        + 3 / 5
        + 3 / 6
        + 1 / 2
        + 4 / 5
        + 2 / 6
        + 1 / 1
        + 2 / 5
    ) / 10
    assert results[0].macro_fact_recall_at_k == pytest.approx(
        expected_macro_recall
    )
    assert results[2].mean_context_precision_at_k == pytest.approx(0.20)
    assert len(results[2].question_results) == len(resolved.questions)


def test_runner_rejects_stale_gold_before_retrieval(tmp_path: Path) -> None:
    stale_gold_path = tmp_path / "stale_gold.json"
    stale_gold_path.write_bytes(GOLD_PATH.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Gold file hash does not match"):
        load_eval_inputs(
            gold_path=stale_gold_path,
            resolved_path=RESOLVED_PATH,
        )


def test_runner_rejects_stale_chunk_index(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    resolved_path = tmp_path / "resolved.json"
    chunks_path = tmp_path / "data/index/MSFT_2023_chunks.json"
    resolved_payload = ResolvedEvalArtifact.model_validate_json(
        RESOLVED_PATH.read_bytes()
    ).model_dump(mode="json")
    resolved_question_ids = {
        question["question_id"] for question in resolved_payload["questions"]
    }
    gold_payload = json.loads(GOLD_PATH.read_bytes())
    gold_payload["questions"] = [
        question
        for question in gold_payload["questions"]
        if question["id"] in resolved_question_ids
    ]
    gold_bytes = json.dumps(
        gold_payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    resolved_payload["gold_source"]["sha256"] = sha256_bytes(gold_bytes)
    gold_path.write_bytes(gold_bytes)
    resolved_path.write_text(json.dumps(resolved_payload), encoding="utf-8")
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Chunk index hash does not match"):
        load_eval_inputs(
            gold_path=gold_path,
            resolved_path=resolved_path,
            repository_root=tmp_path,
        )


def test_cutoffs_are_positive_unique_and_sorted() -> None:
    assert normalize_cutoffs([5, 1, 3]) == [1, 3, 5]

    with pytest.raises(ValueError, match="positive"):
        normalize_cutoffs([0, 5])

    with pytest.raises(ValueError, match="duplicates"):
        normalize_cutoffs([1, 1])
