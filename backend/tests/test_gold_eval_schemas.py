import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.evals.schemas import GoldEvalDataset


GOLD_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evals"
    / "gold"
    / "msft_2023_questions.json"
)


def load_gold_dataset_dict() -> dict:
    return json.loads(GOLD_DATASET_PATH.read_text(encoding="utf-8"))


def test_cybersecurity_gold_dataset_with_optional_facts_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())

    assert len(dataset.questions) == 10
    question = dataset.questions[0]
    assert question.annotation.review_status == "verified"
    assert len(question.required_facts) == 4
    assert len(question.optional_facts) == 9
    assert "product development delays" in question.required_facts[1].claim


def test_verified_ai_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[1]

    assert question.id == "msft_2023_ai_risks"
    assert question.annotation.review_status == "verified"
    assert len(question.required_facts) == 4
    assert len(question.optional_facts) == 4


def test_verified_business_competition_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[2]

    assert question.id == "msft_2023_business_competition"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 1"]
    assert len(question.required_facts) == 7
    assert len(question.optional_facts) == 6


def test_verified_cloud_infrastructure_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[3]

    assert question.id == "msft_2023_cloud_infrastructure_risks"
    assert question.query == (
        "What operational risks does Microsoft describe in maintaining its "
        "datacenter and cloud infrastructure?"
    )
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 1A"]
    assert len(question.required_facts) == 5
    assert len(question.optional_facts) == 4


def test_verified_economic_conditions_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[4]

    assert question.id == "msft_2023_economic_conditions_mda"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 7"]
    assert len(question.required_facts) == 6
    assert len(question.optional_facts) == 4


def test_verified_foreign_exchange_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[5]

    assert question.id == "msft_2023_foreign_exchange_mda"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 7"]
    assert len(question.required_facts) == 2
    assert len(question.optional_facts) == 5
    assert sum(len(fact.evidence_units) for fact in question.optional_facts) == 6


def test_verified_revenue_growth_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[6]

    assert question.id == "msft_2023_revenue_growth_mda"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 7"]
    assert len(question.required_facts) == 5
    assert len(question.optional_facts) == 4
    assert sum(len(fact.evidence_units) for fact in question.required_facts) == 12


def test_verified_revenue_recognition_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[7]

    assert question.id == "msft_2023_revenue_recognition"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 8"]
    assert len(question.required_facts) == 6
    assert len(question.optional_facts) == 6
    assert sum(len(fact.evidence_units) for fact in question.optional_facts) == 8


def test_verified_derivatives_market_risk_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[8]

    assert question.id == "msft_2023_derivatives_market_risk"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 7A"]
    assert len(question.required_facts) == 1
    assert len(question.optional_facts) == 5


def test_verified_liquidity_cash_flows_gold_question_is_valid() -> None:
    dataset = GoldEvalDataset.model_validate(load_gold_dataset_dict())
    question = dataset.questions[9]

    assert question.id == "msft_2023_liquidity_cash_flows_mda"
    assert question.annotation.review_status == "verified"
    assert question.scope.sections == ["Item 7"]
    assert len(question.required_facts) == 5
    assert len(question.optional_facts) == 8


def test_gold_dataset_rejects_resolver_generated_fields() -> None:
    raw_dataset = load_gold_dataset_dict()
    evidence = raw_dataset["questions"][0]["required_facts"][0]["evidence_units"][0]
    evidence["quote_sha256"] = "generated-value-does-not-belong-in-gold"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GoldEvalDataset.model_validate(raw_dataset)


def test_gold_dataset_rejects_evidence_outside_question_scope() -> None:
    raw_dataset = load_gold_dataset_dict()
    evidence = raw_dataset["questions"][0]["required_facts"][0]["evidence_units"][0]
    evidence["section"] = "Item 7"

    with pytest.raises(ValidationError, match="outside question scope"):
        GoldEvalDataset.model_validate(raw_dataset)


def test_gold_dataset_rejects_duplicate_fact_ids() -> None:
    raw_dataset = load_gold_dataset_dict()
    facts = raw_dataset["questions"][0]["required_facts"]
    facts[1]["fact_id"] = facts[0]["fact_id"]

    with pytest.raises(ValidationError, match="fact IDs must be unique"):
        GoldEvalDataset.model_validate(raw_dataset)


def test_gold_dataset_rejects_duplicate_ids_across_required_and_optional_facts() -> None:
    raw_dataset = load_gold_dataset_dict()
    question = raw_dataset["questions"][0]
    question["optional_facts"][0]["fact_id"] = question["required_facts"][0][
        "fact_id"
    ]

    with pytest.raises(ValidationError, match="fact IDs must be unique"):
        GoldEvalDataset.model_validate(raw_dataset)
