import hashlib

from backend.schemas import DocumentChunk, FilingSection


DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 64
CHARS_PER_TOKEN_ESTIMATE = 4


def chunk_filing_sections(
    sections: list[FilingSection],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[DocumentChunk]:
    """Chunk filing sections into retrieval-ready DocumentChunk objects."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative.")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    chunks: list[DocumentChunk] = []

    for section in sections:
        section_chunks = chunk_single_section(
            section=section,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(section_chunks)

    return chunks


def chunk_single_section(
    section: FilingSection,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[DocumentChunk]:
    """Chunk one filing section without crossing section boundaries."""
    max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN_ESTIMATE
    step_chars = max_chars - overlap_chars

    text = section.text.strip()

    if not text:
        return []

    chunks: list[DocumentChunk] = []
    local_start = 0
    chunk_index = 0

    while local_start < len(text):
        local_start = _move_start_to_boundary(text=text, proposed_start=local_start)
        local_end = min(local_start + max_chars, len(text))

        if local_end < len(text):
            local_end = _move_end_to_boundary(
                text=text,
                start=local_start,
                proposed_end=local_end,
            )

        trimmed_start, trimmed_end = _trim_span(text=text, start=local_start, end=local_end)
        chunk_text = text[trimmed_start:trimmed_end]

        if chunk_text:
            absolute_start = section.char_start + trimmed_start
            absolute_end = section.char_start + trimmed_end
            token_count = estimate_token_count(chunk_text)

            chunks.append(
                DocumentChunk(
                    chunk_id=build_chunk_id(
                        ticker=section.metadata.ticker,
                        fiscal_year=section.metadata.fiscal_year,
                        accession_number=section.metadata.accession_number,
                        section=section.section,
                        chunk_index=chunk_index,
                    ),
                    metadata=section.metadata,
                    section=section.section,
                    section_title=section.section_title,
                    chunk_type="text",
                    text=chunk_text,
                    char_start=absolute_start,
                    char_end=absolute_end,
                    token_count=token_count,
                )
            )

            chunk_index += 1

        if local_end >= len(text):
            break

        next_start = max(local_end - overlap_chars, local_start + step_chars)
        if next_start <= local_start:
            next_start = local_start + 1

        local_start = next_start

    return chunks


def _move_start_to_boundary(text: str, proposed_start: int) -> int:
    """Move chunk start forward if it lands in the middle of a word."""
    if proposed_start <= 0:
        return 0

    if proposed_start >= len(text):
        return len(text)

    start = proposed_start

    while start < len(text) and text[start].isspace():
        start += 1

    if start >= len(text):
        return len(text)

    previous_char = text[start - 1]
    current_char = text[start]

    if not (previous_char.isalnum() and current_char.isalnum()):
        return start

    search_limit = min(start + 120, len(text))

    while start < search_limit and text[start].isalnum():
        start += 1

    while start < len(text) and text[start].isspace():
        start += 1

    if start >= search_limit:
        return proposed_start

    return start


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace while preserving accurate offsets."""
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def _move_end_to_boundary(text: str, start: int, proposed_end: int) -> int:
    """Move chunk end backward to a nearby sentence or newline boundary."""
    search_window = text[start:proposed_end]
    boundary_candidates = [
        search_window.rfind("\n"),
        search_window.rfind(". "),
        search_window.rfind("; "),
    ]

    best_boundary = max(boundary_candidates)

    minimum_chunk_chars = 800

    if best_boundary >= minimum_chunk_chars:
        return start + best_boundary + 1

    return proposed_end


def estimate_token_count(text: str) -> int:
    """Estimate token count without requiring tokenizer dependencies."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def build_chunk_id(
    ticker: str,
    fiscal_year: int,
    accession_number: str,
    section: str,
    chunk_index: int,
) -> str:
    """Build a stable chunk ID from filing identity and chunk position."""
    raw_id = f"{ticker}:{fiscal_year}:{accession_number}:{section}:{chunk_index}"
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]

    return f"{ticker}_{fiscal_year}_{section.replace(' ', '')}_{chunk_index}_{digest}"