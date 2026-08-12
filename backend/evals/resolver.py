import hashlib
import json

from backend.evals.schemas import (
    EvidenceResolution,
    FactResolution,
    GoldEvalDataset,
    GoldFact,
    QuestionResolution,
    QuestionResolutionValidation,
    ResolutionValidationSummary,
    ResolvedEvalArtifact,
    ResolvedFilingSource,
    ResolvedGoldSource,
    ResolvedIndexSource,
    ResolvedSectionSource,
    ResolverMetadata,
)
from backend.parsing.section_extractor import extract_filing_sections
from backend.schemas import DocumentChunk, FilingMetadata, FilingSection, RawFiling


class ResolutionError(ValueError):
    """Raised when gold evidence cannot be resolved unambiguously."""


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact input bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of exact UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def resolve_eval_artifact(
    *,
    gold_bytes: bytes,
    raw_filing_bytes: bytes,
    chunks_bytes: bytes,
    gold_path: str,
    raw_filing_path: str,
    chunks_path: str,
    index_name: str,
    resolution_name: str,
) -> ResolvedEvalArtifact:
    """Resolve gold evidence to an exact filing and chunk index without file I/O."""
    gold_dataset = _parse_gold_dataset(gold_bytes)
    chunks = _parse_chunks(chunks_bytes)
    filing_metadata = chunks[0].metadata

    _validate_filing_identity(gold_dataset, filing_metadata)
    _validate_chunk_index(chunks, filing_metadata)

    unverified_question_ids = [
        question.id
        for question in gold_dataset.questions
        if question.annotation.review_status != "verified"
    ]
    if unverified_question_ids:
        raise ResolutionError(
            "All gold questions must be verified before resolution: "
            + ", ".join(unverified_question_ids)
        )

    try:
        raw_content = raw_filing_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResolutionError("Raw filing is not valid UTF-8.") from exc

    raw_filing = RawFiling(
        metadata=filing_metadata,
        content=raw_content,
        content_type="html",
    )

    required_sections = _ordered_unique(
        [
            section
            for question in gold_dataset.questions
            for section in question.scope.sections
        ]
    )
    sections = extract_filing_sections(
        raw_filing,
        target_sections=tuple(required_sections),
    )
    sections_by_name = {section.section: section for section in sections}

    missing_sections = [
        section for section in required_sections if section not in sections_by_name
    ]
    if missing_sections:
        raise ResolutionError(
            "Could not extract required filing sections: "
            + ", ".join(missing_sections)
        )

    section_sources = [
        ResolvedSectionSource(
            section=section_name,
            document_char_start=sections_by_name[section_name].char_start,
            document_char_end=(
                sections_by_name[section_name].char_start
                + len(sections_by_name[section_name].text)
            ),
            normalized_text_sha256=sha256_text(
                sections_by_name[section_name].text
            ),
        )
        for section_name in required_sections
    ]

    question_resolutions: list[QuestionResolution] = []

    for question in gold_dataset.questions:
        required_fact_resolutions = _resolve_facts(
            facts=question.required_facts,
            sections_by_name=sections_by_name,
            chunks=chunks,
        )
        optional_fact_resolutions = _resolve_facts(
            facts=question.optional_facts,
            sections_by_name=sections_by_name,
            chunks=chunks,
        )
        required_evidence_chunk_ids = _ordered_unique(
            [
                chunk_id
                for resolution in required_fact_resolutions
                for chunk_id in resolution.resolved_chunk_ids
            ]
        )
        relevant_context_chunk_ids = _ordered_unique(
            [
                chunk_id
                for resolution in (
                    required_fact_resolutions + optional_fact_resolutions
                )
                for chunk_id in resolution.resolved_chunk_ids
            ]
        )
        evidence_count = sum(
            len(resolution.evidence_resolutions)
            for resolution in (
                required_fact_resolutions + optional_fact_resolutions
            )
        )
        question_resolutions.append(
            QuestionResolution(
                question_id=question.id,
                required_fact_resolutions=required_fact_resolutions,
                optional_fact_resolutions=optional_fact_resolutions,
                required_evidence_chunk_ids=required_evidence_chunk_ids,
                relevant_context_chunk_ids=relevant_context_chunk_ids,
                validation=QuestionResolutionValidation(
                    required_fact_count=len(required_fact_resolutions),
                    resolved_required_fact_count=len(required_fact_resolutions),
                    optional_fact_count=len(optional_fact_resolutions),
                    resolved_optional_fact_count=len(optional_fact_resolutions),
                    evidence_count=evidence_count,
                    resolved_evidence_count=evidence_count,
                    status="passed",
                ),
            )
        )

    required_fact_count = sum(
        len(question.required_fact_resolutions)
        for question in question_resolutions
    )
    optional_fact_count = sum(
        len(question.optional_fact_resolutions)
        for question in question_resolutions
    )
    evidence_count = sum(
        len(fact.evidence_resolutions)
        for question in question_resolutions
        for fact in (
            question.required_fact_resolutions
            + question.optional_fact_resolutions
        )
    )

    return ResolvedEvalArtifact(
        schema_version="1.0",
        artifact_type="resolved_retrieval_gold",
        resolution_name=resolution_name,
        resolver=ResolverMetadata(
            name="exact_quote_chunk_resolver",
            version="1.0",
            quote_match_policy="unique_exact_match",
            chunk_match_policy="full_quote_containment",
        ),
        gold_source=ResolvedGoldSource(
            dataset_name=gold_dataset.dataset_name,
            path=gold_path,
            sha256=sha256_bytes(gold_bytes),
        ),
        filing_source=ResolvedFilingSource(
            ticker=gold_dataset.filing.ticker,
            fiscal_year=gold_dataset.filing.fiscal_year,
            filing_type=gold_dataset.filing.filing_type,
            accession_number=gold_dataset.filing.accession_number,
            raw_path=raw_filing_path,
            raw_sha256=sha256_bytes(raw_filing_bytes),
            normalization_method="clean_filing_html_v1",
            sections=section_sources,
        ),
        index_source=ResolvedIndexSource(
            index_name=index_name,
            chunks_path=chunks_path,
            chunks_sha256=sha256_bytes(chunks_bytes),
            chunk_count=len(chunks),
        ),
        questions=question_resolutions,
        validation_summary=ResolutionValidationSummary(
            question_count=len(question_resolutions),
            required_fact_count=required_fact_count,
            optional_fact_count=optional_fact_count,
            evidence_count=evidence_count,
            unresolved_evidence_ids=[],
            status="passed",
        ),
    )


def _resolve_facts(
    *,
    facts: list[GoldFact],
    sections_by_name: dict[str, FilingSection],
    chunks: list[DocumentChunk],
) -> list[FactResolution]:
    fact_resolutions: list[FactResolution] = []

    for fact in facts:
        evidence_resolutions: list[EvidenceResolution] = []

        for evidence in fact.evidence_units:
            section = sections_by_name[evidence.section]
            match_starts = _find_exact_match_starts(section.text, evidence.quote)

            if len(match_starts) != 1:
                raise ResolutionError(
                    f"Evidence {evidence.evidence_id!r} must appear exactly once "
                    f"in {evidence.section}; found {len(match_starts)} matches."
                )

            section_char_start = match_starts[0]
            section_char_end = section_char_start + len(evidence.quote)
            document_char_start = section.char_start + section_char_start
            document_char_end = section.char_start + section_char_end

            matched_chunk_ids = [
                chunk.chunk_id
                for chunk in chunks
                if chunk.section == evidence.section
                and chunk.char_start <= document_char_start
                and chunk.char_end >= document_char_end
                and evidence.quote in chunk.text
            ]
            if not matched_chunk_ids:
                raise ResolutionError(
                    f"No chunk fully contains evidence {evidence.evidence_id!r}."
                )

            evidence_resolutions.append(
                EvidenceResolution(
                    evidence_id=evidence.evidence_id,
                    section=evidence.section,
                    quote_sha256=sha256_text(evidence.quote),
                    section_char_start=section_char_start,
                    section_char_end=section_char_end,
                    document_char_start=document_char_start,
                    document_char_end=document_char_end,
                    exact_match_count=1,
                    match_method="exact",
                    matched_chunk_ids=matched_chunk_ids,
                )
            )

        resolved_chunk_ids = _ordered_unique(
            [
                chunk_id
                for resolution in evidence_resolutions
                for chunk_id in resolution.matched_chunk_ids
            ]
        )
        fact_resolutions.append(
            FactResolution(
                fact_id=fact.fact_id,
                evidence_resolutions=evidence_resolutions,
                resolved_chunk_ids=resolved_chunk_ids,
            )
        )

    return fact_resolutions


def _parse_gold_dataset(content: bytes) -> GoldEvalDataset:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResolutionError("Gold dataset is not valid UTF-8 JSON.") from exc
    return GoldEvalDataset.model_validate(payload)


def _parse_chunks(content: bytes) -> list[DocumentChunk]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResolutionError("Chunk index is not valid UTF-8 JSON.") from exc

    if not isinstance(payload, list) or not payload:
        raise ResolutionError("Chunk index must be a non-empty JSON list.")
    return [DocumentChunk.model_validate(chunk) for chunk in payload]


def _validate_filing_identity(
    gold_dataset: GoldEvalDataset,
    filing_metadata: FilingMetadata,
) -> None:
    expected = gold_dataset.filing
    identity_pairs = {
        "ticker": (expected.ticker, filing_metadata.ticker),
        "fiscal_year": (expected.fiscal_year, filing_metadata.fiscal_year),
        "filing_type": (expected.filing_type, filing_metadata.filing_type),
        "accession_number": (
            expected.accession_number,
            filing_metadata.accession_number,
        ),
    }
    mismatches = [
        f"{field}: gold={gold_value!r}, index={index_value!r}"
        for field, (gold_value, index_value) in identity_pairs.items()
        if gold_value != index_value
    ]
    if mismatches:
        raise ResolutionError("Filing identity mismatch: " + "; ".join(mismatches))


def _validate_chunk_index(
    chunks: list[DocumentChunk],
    filing_metadata: FilingMetadata,
) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ResolutionError("Chunk index contains duplicate chunk IDs.")

    for chunk in chunks:
        if chunk.metadata != filing_metadata:
            raise ResolutionError(
                f"Chunk {chunk.chunk_id!r} has inconsistent filing metadata."
            )
        if chunk.char_end - chunk.char_start != len(chunk.text):
            raise ResolutionError(
                f"Chunk {chunk.chunk_id!r} has inconsistent text offsets."
            )


def _find_exact_match_starts(text: str, quote: str) -> list[int]:
    match_starts: list[int] = []
    search_start = 0

    while True:
        match_start = text.find(quote, search_start)
        if match_start < 0:
            return match_starts
        match_starts.append(match_start)
        search_start = match_start + 1


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
