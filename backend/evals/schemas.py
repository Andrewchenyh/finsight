from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.schemas import FilingSectionName, FilingType, RetrievalMode


Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _ordered_unique(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving their first-seen order."""
    return list(dict.fromkeys(values))


def _require_relative_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact paths must be repository-relative.")
    return value


class StrictEvalModel(BaseModel):
    """Base model that rejects undocumented evaluation fields."""

    model_config = ConfigDict(extra="forbid")


class GoldFilingIdentity(StrictEvalModel):
    """Logical identity of the filing annotated by a gold dataset."""

    ticker: str = Field(..., min_length=1)
    fiscal_year: int = Field(..., ge=1994)
    filing_type: FilingType
    accession_number: str = Field(
        ...,
        pattern=r"^\d{10}-\d{2}-\d{6}$",
        description="SEC accession number in dashed form.",
    )

    @field_validator("ticker")
    @classmethod
    def require_uppercase_ticker(cls, value: str) -> str:
        if value != value.upper():
            raise ValueError("ticker must be uppercase.")
        return value


class GoldQuestionScope(StrictEvalModel):
    """Filing sections a question and its evidence are allowed to use."""

    sections: list[FilingSectionName] = Field(..., min_length=1)

    @field_validator("sections")
    @classmethod
    def require_unique_known_sections(
        cls,
        value: list[FilingSectionName],
    ) -> list[FilingSectionName]:
        if "Unknown" in value:
            raise ValueError("gold question scope cannot contain Unknown.")
        if len(value) != len(set(value)):
            raise ValueError("gold question scope sections must be unique.")
        return value


class GoldEvidenceUnit(StrictEvalModel):
    """One verbatim filing passage that supports a required fact."""

    evidence_id: str = Field(..., min_length=1)
    section: FilingSectionName
    quote: str = Field(..., min_length=1)

    @field_validator("section")
    @classmethod
    def require_known_section(cls, value: FilingSectionName) -> FilingSectionName:
        if value == "Unknown":
            raise ValueError("gold evidence section cannot be Unknown.")
        return value

    @field_validator("quote")
    @classmethod
    def preserve_exact_quote_boundaries(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(
                "gold evidence quote must not have leading or trailing whitespace."
            )
        return value


class GoldFact(StrictEvalModel):
    """One answer-level claim and its acceptable source evidence."""

    fact_id: str = Field(..., min_length=1)
    claim: str = Field(..., min_length=1)
    evidence_units: list[GoldEvidenceUnit] = Field(..., min_length=1)

    @model_validator(mode="after")
    def require_unique_evidence_ids(self) -> GoldFact:
        evidence_ids = [evidence.evidence_id for evidence in self.evidence_units]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a required fact.")
        return self


class GoldAnnotation(StrictEvalModel):
    """Human provenance and review state for one gold question."""

    method: Literal["manual_source_review"]
    annotator: str = Field(..., min_length=1)
    review_status: Literal["draft", "verified"]


class GoldQuestion(StrictEvalModel):
    """One source-grounded retrieval evaluation question."""

    id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    scope: GoldQuestionScope
    reference_answer: str = Field(..., min_length=1)
    required_facts: list[GoldFact] = Field(..., min_length=1)
    optional_facts: list[GoldFact] = Field(default_factory=list)
    annotation: GoldAnnotation

    @model_validator(mode="after")
    def validate_fact_and_evidence_identity(self) -> GoldQuestion:
        all_facts = self.required_facts + self.optional_facts
        fact_ids = [fact.fact_id for fact in all_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact IDs must be unique within a question.")

        scoped_sections = set(self.scope.sections)
        evidence_ids: list[str] = []

        for fact in all_facts:
            for evidence in fact.evidence_units:
                evidence_ids.append(evidence.evidence_id)
                if evidence.section not in scoped_sections:
                    raise ValueError(
                        f"evidence section {evidence.section!r} is outside question scope."
                    )

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a question.")

        return self


class GoldEvalDataset(StrictEvalModel):
    """Human-maintained, source-grounded retrieval evaluation dataset."""

    schema_version: Literal["1.0"]
    dataset_name: str = Field(..., min_length=1)
    filing: GoldFilingIdentity
    questions: list[GoldQuestion] = Field(..., min_length=1)

    @model_validator(mode="after")
    def require_unique_question_ids(self) -> GoldEvalDataset:
        question_ids = [question.id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question IDs must be unique within a gold dataset.")
        return self


class ResolverMetadata(StrictEvalModel):
    """Versioned policies used to generate a resolved artifact."""

    name: Literal["exact_quote_chunk_resolver"]
    version: Literal["1.0"]
    quote_match_policy: Literal["unique_exact_match"]
    chunk_match_policy: Literal["full_quote_containment"]


class ResolvedGoldSource(StrictEvalModel):
    """Exact human-maintained gold file used as resolver input."""

    dataset_name: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    sha256: Sha256Hex

    _validate_path = field_validator("path")(_require_relative_repo_path)


class ResolvedSectionSource(StrictEvalModel):
    """Normalized filing section searched for gold evidence."""

    section: FilingSectionName
    document_char_start: int = Field(..., ge=0)
    document_char_end: int = Field(..., ge=1)
    normalized_text_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_section(self) -> ResolvedSectionSource:
        if self.section == "Unknown":
            raise ValueError("resolved filing section cannot be Unknown.")
        if self.document_char_end <= self.document_char_start:
            raise ValueError("section document_char_end must be greater than its start.")
        return self


class ResolvedFilingSource(GoldFilingIdentity):
    """Exact raw and normalized filing used during resolution."""

    raw_path: str = Field(..., min_length=1)
    raw_sha256: Sha256Hex
    normalization_method: Literal["clean_filing_html_v1"]
    sections: list[ResolvedSectionSource] = Field(..., min_length=1)

    _validate_raw_path = field_validator("raw_path")(_require_relative_repo_path)

    @model_validator(mode="after")
    def require_unique_sections(self) -> ResolvedFilingSource:
        section_names = [section.section for section in self.sections]
        if len(section_names) != len(set(section_names)):
            raise ValueError("resolved filing sections must be unique.")
        return self


class ResolvedIndexSource(StrictEvalModel):
    """Exact chunk index to which source evidence was mapped."""

    index_name: str = Field(..., min_length=1)
    chunks_path: str = Field(..., min_length=1)
    chunks_sha256: Sha256Hex
    chunk_count: int = Field(..., ge=1)

    _validate_chunks_path = field_validator("chunks_path")(
        _require_relative_repo_path
    )


class EvidenceResolution(StrictEvalModel):
    """Automatically computed source span and chunk mapping for one quote."""

    evidence_id: str = Field(..., min_length=1)
    section: FilingSectionName
    quote_sha256: Sha256Hex
    section_char_start: int = Field(..., ge=0)
    section_char_end: int = Field(..., ge=1)
    document_char_start: int = Field(..., ge=0)
    document_char_end: int = Field(..., ge=1)
    exact_match_count: Literal[1]
    match_method: Literal["exact"]
    matched_chunk_ids: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_spans_and_chunks(self) -> EvidenceResolution:
        if self.section == "Unknown":
            raise ValueError("resolved evidence section cannot be Unknown.")
        if self.section_char_end <= self.section_char_start:
            raise ValueError("section_char_end must be greater than section_char_start.")
        if self.document_char_end <= self.document_char_start:
            raise ValueError("document_char_end must be greater than document_char_start.")

        section_span_length = self.section_char_end - self.section_char_start
        document_span_length = self.document_char_end - self.document_char_start
        if section_span_length != document_span_length:
            raise ValueError("section and document evidence spans must have equal lengths.")

        if len(self.matched_chunk_ids) != len(set(self.matched_chunk_ids)):
            raise ValueError("matched chunk IDs must be unique within evidence.")
        return self


class FactResolution(StrictEvalModel):
    """Generated evidence and chunk mappings for one gold fact."""

    fact_id: str = Field(..., min_length=1)
    evidence_resolutions: list[EvidenceResolution] = Field(..., min_length=1)
    resolved_chunk_ids: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_resolved_chunk_union(self) -> FactResolution:
        evidence_ids = [
            evidence.evidence_id for evidence in self.evidence_resolutions
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a fact resolution.")

        expected_chunk_ids = _ordered_unique(
            [
                chunk_id
                for evidence in self.evidence_resolutions
                for chunk_id in evidence.matched_chunk_ids
            ]
        )
        if self.resolved_chunk_ids != expected_chunk_ids:
            raise ValueError(
                "resolved_chunk_ids must equal the ordered evidence chunk union."
            )
        return self


class QuestionResolutionValidation(StrictEvalModel):
    """Generated completeness counts for one resolved question."""

    required_fact_count: int = Field(..., ge=1)
    resolved_required_fact_count: int = Field(..., ge=1)
    optional_fact_count: int = Field(..., ge=0)
    resolved_optional_fact_count: int = Field(..., ge=0)
    evidence_count: int = Field(..., ge=1)
    resolved_evidence_count: int = Field(..., ge=1)
    status: Literal["passed"]


class QuestionResolution(StrictEvalModel):
    """Generated required/optional mappings for one question."""

    question_id: str = Field(..., min_length=1)
    required_fact_resolutions: list[FactResolution] = Field(..., min_length=1)
    optional_fact_resolutions: list[FactResolution]
    required_evidence_chunk_ids: list[str] = Field(..., min_length=1)
    relevant_context_chunk_ids: list[str] = Field(..., min_length=1)
    validation: QuestionResolutionValidation

    @model_validator(mode="after")
    def validate_question_resolution(self) -> QuestionResolution:
        all_fact_resolutions = (
            self.required_fact_resolutions + self.optional_fact_resolutions
        )
        fact_ids = [fact.fact_id for fact in all_fact_resolutions]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact IDs must be unique within a question resolution.")

        evidence_ids = [
            evidence.evidence_id
            for fact in all_fact_resolutions
            for evidence in fact.evidence_resolutions
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a question resolution.")

        required_evidence_chunk_ids = _ordered_unique(
            [
                chunk_id
                for fact in self.required_fact_resolutions
                for chunk_id in fact.resolved_chunk_ids
            ]
        )
        if self.required_evidence_chunk_ids != required_evidence_chunk_ids:
            raise ValueError(
                "required_evidence_chunk_ids must equal the ordered required-fact chunk union."
            )

        relevant_context_chunk_ids = _ordered_unique(
            [
                chunk_id
                for fact in all_fact_resolutions
                for chunk_id in fact.resolved_chunk_ids
            ]
        )
        if self.relevant_context_chunk_ids != relevant_context_chunk_ids:
            raise ValueError(
                "relevant_context_chunk_ids must equal the ordered all-fact chunk union."
            )

        required_fact_count = len(self.required_fact_resolutions)
        optional_fact_count = len(self.optional_fact_resolutions)
        evidence_count = len(evidence_ids)
        if self.validation.required_fact_count != required_fact_count:
            raise ValueError(
                "required_fact_count does not match required fact resolutions."
            )
        if self.validation.resolved_required_fact_count != required_fact_count:
            raise ValueError(
                "resolved_required_fact_count does not match required fact resolutions."
            )
        if self.validation.optional_fact_count != optional_fact_count:
            raise ValueError(
                "optional_fact_count does not match optional fact resolutions."
            )
        if self.validation.resolved_optional_fact_count != optional_fact_count:
            raise ValueError(
                "resolved_optional_fact_count does not match optional fact resolutions."
            )
        if self.validation.evidence_count != evidence_count:
            raise ValueError("evidence_count does not match evidence resolutions.")
        if self.validation.resolved_evidence_count != evidence_count:
            raise ValueError(
                "resolved_evidence_count does not match evidence resolutions."
            )
        return self


class ResolutionValidationSummary(StrictEvalModel):
    """Generated completeness totals for the resolved artifact."""

    question_count: int = Field(..., ge=1)
    required_fact_count: int = Field(..., ge=1)
    optional_fact_count: int = Field(..., ge=0)
    evidence_count: int = Field(..., ge=1)
    unresolved_evidence_ids: list[str]
    status: Literal["passed"]

    @field_validator("unresolved_evidence_ids")
    @classmethod
    def require_no_unresolved_evidence(cls, value: list[str]) -> list[str]:
        if value:
            raise ValueError("a passed resolved artifact cannot contain unresolved evidence.")
        return value


class ResolvedEvalArtifact(StrictEvalModel):
    """Deterministic mapping from human gold evidence to one exact chunk index."""

    schema_version: Literal["1.0"]
    artifact_type: Literal["resolved_retrieval_gold"]
    resolution_name: str = Field(..., min_length=1)
    resolver: ResolverMetadata
    gold_source: ResolvedGoldSource
    filing_source: ResolvedFilingSource
    index_source: ResolvedIndexSource
    questions: list[QuestionResolution] = Field(..., min_length=1)
    validation_summary: ResolutionValidationSummary

    @model_validator(mode="after")
    def validate_artifact_totals_and_offsets(self) -> ResolvedEvalArtifact:
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question IDs must be unique within a resolved artifact.")

        section_sources = {
            section.section: section for section in self.filing_source.sections
        }
        for question in self.questions:
            all_fact_resolutions = (
                question.required_fact_resolutions
                + question.optional_fact_resolutions
            )
            for fact in all_fact_resolutions:
                for evidence in fact.evidence_resolutions:
                    section_source = section_sources.get(evidence.section)
                    if section_source is None:
                        raise ValueError(
                            f"evidence section {evidence.section!r} is absent from filing source."
                        )

                    expected_document_start = (
                        section_source.document_char_start
                        + evidence.section_char_start
                    )
                    expected_document_end = (
                        section_source.document_char_start + evidence.section_char_end
                    )
                    if (
                        evidence.document_char_start != expected_document_start
                        or evidence.document_char_end != expected_document_end
                    ):
                        raise ValueError(
                            "evidence document offsets do not match section offsets."
                        )
                    if evidence.document_char_end > section_source.document_char_end:
                        raise ValueError("evidence span exceeds its resolved section.")

        required_fact_count = sum(
            len(question.required_fact_resolutions) for question in self.questions
        )
        optional_fact_count = sum(
            len(question.optional_fact_resolutions) for question in self.questions
        )
        evidence_count = sum(
            len(fact.evidence_resolutions)
            for question in self.questions
            for fact in (
                question.required_fact_resolutions
                + question.optional_fact_resolutions
            )
        )
        if self.validation_summary.question_count != len(self.questions):
            raise ValueError("summary question_count does not match questions.")
        if self.validation_summary.required_fact_count != required_fact_count:
            raise ValueError(
                "summary required_fact_count does not match fact resolutions."
            )
        if self.validation_summary.optional_fact_count != optional_fact_count:
            raise ValueError(
                "summary optional_fact_count does not match fact resolutions."
            )
        if self.validation_summary.evidence_count != evidence_count:
            raise ValueError("summary evidence_count does not match evidence resolutions.")
        return self


class RetrievalQuestionMetrics(StrictEvalModel):
    """Source-grounded retrieval metrics for one question at one cutoff."""

    question_id: str = Field(..., min_length=1)
    top_k: int = Field(..., ge=1)
    retrieved_chunk_ids: list[str]
    required_evidence_chunk_ids: list[str] = Field(..., min_length=1)
    relevant_context_chunk_ids: list[str] = Field(..., min_length=1)
    relevant_context_retrieved_chunk_ids: list[str]
    covered_fact_ids: list[str]
    required_fact_count: int = Field(..., ge=1)
    covered_fact_count: int = Field(..., ge=0)
    required_fact_hit_at_k: bool
    fact_recall_at_k: float = Field(..., ge=0.0, le=1.0)
    full_coverage_at_k: bool
    context_precision_at_k: float = Field(..., ge=0.0, le=1.0)
    first_required_evidence_rank: int | None = Field(default=None, ge=1)
    required_evidence_reciprocal_rank_at_k: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_metric_relationships(self) -> RetrievalQuestionMetrics:
        if len(self.retrieved_chunk_ids) > self.top_k:
            raise ValueError("retrieved_chunk_ids cannot contain more than top_k items.")
        if len(self.retrieved_chunk_ids) != len(set(self.retrieved_chunk_ids)):
            raise ValueError("retrieved chunk IDs must be unique.")
        if len(self.required_evidence_chunk_ids) != len(
            set(self.required_evidence_chunk_ids)
        ):
            raise ValueError("required evidence chunk IDs must be unique.")
        if len(self.relevant_context_chunk_ids) != len(
            set(self.relevant_context_chunk_ids)
        ):
            raise ValueError("relevant context chunk IDs must be unique.")
        if not set(self.required_evidence_chunk_ids).issubset(
            self.relevant_context_chunk_ids
        ):
            raise ValueError(
                "required evidence chunks must be a subset of relevant context chunks."
            )
        if len(self.relevant_context_retrieved_chunk_ids) != len(
            set(self.relevant_context_retrieved_chunk_ids)
        ):
            raise ValueError("relevant context retrieved chunk IDs must be unique.")
        if len(self.covered_fact_ids) != len(set(self.covered_fact_ids)):
            raise ValueError("covered fact IDs must be unique.")

        relevant_context_ids = [
            chunk_id
            for chunk_id in self.retrieved_chunk_ids
            if chunk_id in set(self.relevant_context_chunk_ids)
        ]
        if self.relevant_context_retrieved_chunk_ids != relevant_context_ids:
            raise ValueError(
                "relevant_context_retrieved_chunk_ids must be the ordered context intersection."
            )
        if self.covered_fact_count != len(self.covered_fact_ids):
            raise ValueError("covered_fact_count does not match covered_fact_ids.")
        if self.covered_fact_count > self.required_fact_count:
            raise ValueError("covered_fact_count cannot exceed required_fact_count.")

        expected_required_fact_hit = self.covered_fact_count > 0
        if self.required_fact_hit_at_k != expected_required_fact_hit:
            raise ValueError("required_fact_hit_at_k does not match covered facts.")

        expected_fact_recall = self.covered_fact_count / self.required_fact_count
        if not math.isclose(self.fact_recall_at_k, expected_fact_recall):
            raise ValueError("fact_recall_at_k does not match fact counts.")

        expected_full_coverage = self.covered_fact_count == self.required_fact_count
        if self.full_coverage_at_k != expected_full_coverage:
            raise ValueError("full_coverage_at_k does not match fact counts.")

        expected_context_precision = len(relevant_context_ids) / self.top_k
        if not math.isclose(
            self.context_precision_at_k,
            expected_context_precision,
        ):
            raise ValueError(
                "context_precision_at_k does not match relevant chunks and top_k."
            )

        expected_first_required_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(self.retrieved_chunk_ids, start=1)
                if chunk_id in set(self.required_evidence_chunk_ids)
            ),
            None,
        )
        if self.first_required_evidence_rank != expected_first_required_rank:
            raise ValueError(
                "first_required_evidence_rank does not match retrieved chunks."
            )

        expected_reciprocal_rank = (
            1.0 / expected_first_required_rank
            if expected_first_required_rank is not None
            else 0.0
        )
        if not math.isclose(
            self.required_evidence_reciprocal_rank_at_k,
            expected_reciprocal_rank,
        ):
            raise ValueError(
                "required evidence reciprocal rank does not match first required rank."
            )
        return self


class RetrievalRunMetrics(StrictEvalModel):
    """Aggregate source-grounded metrics for one retrieval mode and cutoff."""

    resolution_name: str = Field(..., min_length=1)
    retrieval_mode: RetrievalMode
    top_k: int = Field(..., ge=1)
    question_results: list[RetrievalQuestionMetrics] = Field(..., min_length=1)
    required_fact_hit_rate_at_k: float = Field(..., ge=0.0, le=1.0)
    macro_fact_recall_at_k: float = Field(..., ge=0.0, le=1.0)
    full_coverage_rate_at_k: float = Field(..., ge=0.0, le=1.0)
    mean_context_precision_at_k: float = Field(..., ge=0.0, le=1.0)
    required_evidence_mrr_at_k: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_aggregate_metrics(self) -> RetrievalRunMetrics:
        question_ids = [result.question_id for result in self.question_results]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question IDs must be unique within a metric run.")
        if any(result.top_k != self.top_k for result in self.question_results):
            raise ValueError("all question results must use the run top_k.")

        question_count = len(self.question_results)
        expected_metrics = {
            "required_fact_hit_rate_at_k": sum(
                result.required_fact_hit_at_k for result in self.question_results
            )
            / question_count,
            "macro_fact_recall_at_k": sum(
                result.fact_recall_at_k for result in self.question_results
            )
            / question_count,
            "full_coverage_rate_at_k": sum(
                result.full_coverage_at_k for result in self.question_results
            )
            / question_count,
            "mean_context_precision_at_k": sum(
                result.context_precision_at_k for result in self.question_results
            )
            / question_count,
            "required_evidence_mrr_at_k": sum(
                result.required_evidence_reciprocal_rank_at_k
                for result in self.question_results
            )
            / question_count,
        }
        for field_name, expected_value in expected_metrics.items():
            if not math.isclose(getattr(self, field_name), expected_value):
                raise ValueError(f"{field_name} does not match question results.")
        return self
