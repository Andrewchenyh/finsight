import pytest

from backend.chunking.chunker import chunk_filing_sections
from backend.schemas import FilingMetadata, FilingSection


@pytest.fixture
def sample_metadata() -> FilingMetadata:
    return FilingMetadata(
        company="MICROSOFT CORP",
        ticker="msft",
        cik="0000789019",
        accession_number="0000950170-23-035122",
        filing_type="10-K",
        fiscal_year=2023,
        filing_date="2023-07-27",
        source_url="https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
    )


@pytest.fixture
def sample_sections(sample_metadata: FilingMetadata) -> list[FilingSection]:
    long_text = (
        "Item 1A\n"
        "Microsoft faces intense competition across cloud, productivity, gaming, "
        "and AI infrastructure markets. These risks may affect revenue, margins, "
        "customer adoption, and operating results. "
        * 120
    )

    return [
        FilingSection(
            metadata=sample_metadata,
            section="Item 1A",
            section_title="Risk Factors",
            text=long_text,
            char_start=1_000,
            char_end=1_000 + len(long_text),
        )
    ]


def test_chunk_filing_sections_creates_chunks(sample_sections: list[FilingSection]) -> None:
    chunks = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=120,
        overlap_tokens=20,
    )

    assert len(chunks) > 1


def test_chunks_do_not_exceed_max_token_budget(sample_sections: list[FilingSection]) -> None:
    max_tokens = 120

    chunks = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=max_tokens,
        overlap_tokens=20,
    )

    assert all(chunk.token_count <= max_tokens for chunk in chunks)


def test_chunks_preserve_required_metadata(sample_sections: list[FilingSection]) -> None:
    chunks = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=120,
        overlap_tokens=20,
    )

    first_chunk = chunks[0]

    assert first_chunk.metadata.ticker == "MSFT"
    assert first_chunk.metadata.cik == "789019"
    assert first_chunk.metadata.fiscal_year == 2023
    assert first_chunk.metadata.filing_type == "10-K"
    assert first_chunk.section == "Item 1A"
    assert first_chunk.section_title == "Risk Factors"
    assert first_chunk.chunk_type == "text"
    assert first_chunk.char_start >= sample_sections[0].char_start
    assert first_chunk.char_end <= sample_sections[0].char_end


def test_chunk_ids_are_stable(sample_sections: list[FilingSection]) -> None:
    first_run = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=120,
        overlap_tokens=20,
    )
    second_run = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=120,
        overlap_tokens=20,
    )

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]


def test_chunks_do_not_cross_section_boundaries(sample_sections: list[FilingSection]) -> None:
    section = sample_sections[0]

    chunks = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=120,
        overlap_tokens=20,
    )

    for chunk in chunks:
        assert chunk.char_start >= section.char_start
        assert chunk.char_end <= section.char_end


def test_invalid_max_tokens_raises_error(sample_sections: list[FilingSection]) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        chunk_filing_sections(
            sections=sample_sections,
            max_tokens=0,
            overlap_tokens=20,
        )


def test_negative_overlap_raises_error(sample_sections: list[FilingSection]) -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
        chunk_filing_sections(
            sections=sample_sections,
            max_tokens=120,
            overlap_tokens=-1,
        )


def test_overlap_must_be_smaller_than_max_tokens(sample_sections: list[FilingSection]) -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller than max_tokens"):
        chunk_filing_sections(
            sections=sample_sections,
            max_tokens=120,
            overlap_tokens=120,
        )
        
def test_chunks_do_not_start_in_middle_of_word(sample_sections: list[FilingSection]) -> None:
    chunks = chunk_filing_sections(
        sections=sample_sections,
        max_tokens=120,
        overlap_tokens=20,
    )

    for chunk in chunks[1:]:
        assert not (
            chunk.text[0].isalnum()
            and chunk.char_start > sample_sections[0].char_start
            and sample_sections[0].text[chunk.char_start - sample_sections[0].char_start - 1].isalnum()
        )