from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


FilingType = Literal["10-K", "10-K/A"]
FilingSectionName = Literal[
    "Item 1",
    "Item 1A",
    "Item 1B",
    "Item 2",
    "Item 3",
    "Item 4",
    "Item 5",
    "Item 6",
    "Item 7",
    "Item 7A",
    "Item 8",
    "Item 9",
    "Item 9A",
    "Item 9B",
    "Item 9C",
    "Item 10",
    "Item 11",
    "Item 12",
    "Item 13",
    "Item 14",
    "Item 15",
    "Unknown",
]
ChunkType = Literal["text", "table", "heading"]
RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class FilingMetadata(BaseModel):
    """Identity and source metadata for one SEC filing."""

    company: str = Field(..., min_length=1, description="Company name from SEC metadata.")
    ticker: str = Field(..., min_length=1, description="Public ticker symbol, uppercased.")
    cik: str = Field(..., min_length=1, description="SEC Central Index Key with no punctuation.")
    accession_number: str = Field(
        ...,
        min_length=1,
        description="SEC accession number for the filing.",
    )
    filing_type: FilingType = Field(..., description="SEC filing type.")
    fiscal_year: int = Field(..., ge=1994, description="Fiscal year covered by the filing.")
    filing_date: str = Field(..., description="SEC filing date in YYYY-MM-DD format.")
    source_url: HttpUrl = Field(..., description="Direct URL to the filing document.")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("cik")
    @classmethod
    def normalize_cik(cls, value: str) -> str:
        return value.strip().lstrip("0") or "0"


class RawFiling(BaseModel):
    """Raw filing text/html fetched from SEC before parsing."""

    metadata: FilingMetadata
    content: str = Field(..., min_length=1, description="Raw filing content.")
    content_type: Literal["html", "text"] = Field(
        default="html",
        description="Format of the raw filing content.",
    )


class FilingSection(BaseModel):
    """A major SEC filing section, such as Item 1A or Item 7."""

    metadata: FilingMetadata
    section: FilingSectionName
    section_title: str = Field(..., min_length=1, description="Human-readable section title.")
    text: str = Field(..., min_length=1, description="Cleaned section text.")
    char_start: int = Field(..., ge=0, description="Start character offset in the raw filing.")
    char_end: int = Field(..., ge=0, description="End character offset in the raw filing.")

    @field_validator("char_end")
    @classmethod
    def validate_offsets(cls, value: int, info) -> int:
        char_start = info.data.get("char_start")
        if char_start is not None and value <= char_start:
            raise ValueError("char_end must be greater than char_start.")
        return value


class DocumentChunk(BaseModel):
    """A retrieval-ready chunk created from one filing section."""

    chunk_id: str = Field(..., min_length=1, description="Stable unique ID for this chunk.")
    metadata: FilingMetadata
    section: FilingSectionName
    section_title: str
    chunk_type: ChunkType = "text"
    text: str = Field(..., min_length=1, description="Chunk text sent to the embedding model.")
    char_start: int = Field(..., ge=0, description="Start offset in the raw filing.")
    char_end: int = Field(..., ge=0, description="End offset in the raw filing.")
    token_count: int = Field(..., ge=1, description="Estimated token count for the chunk.")

    @field_validator("char_end")
    @classmethod
    def validate_offsets(cls, value: int, info) -> int:
        char_start = info.data.get("char_start")
        if char_start is not None and value <= char_start:
            raise ValueError("char_end must be greater than char_start.")
        return value


class RetrievalFilter(BaseModel):
    """Optional metadata filters for SEC retrieval."""

    ticker: str | None = Field(default=None, description="Restrict retrieval to one ticker.")
    fiscal_year: int | None = Field(default=None, ge=1994, description="Restrict to one fiscal year.")
    section: FilingSectionName | None = Field(default=None, description="Restrict to one filing section.")
    filing_type: FilingType | None = Field(default=None, description="Restrict to one filing type.")

    @field_validator("ticker")
    @classmethod
    def normalize_optional_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().upper()


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with ranking information attached."""

    chunk: DocumentChunk
    score: float = Field(..., description="Similarity, fusion, or reranking score.")
    rank: int = Field(..., ge=1, description="One-based rank in the retrieval result.")
    retrieval_method: Literal["dense", "bm25", "hybrid", "rerank"] = Field(
        ...,
        description="Retrieval stage that produced this ranking.",
    )


class SourceCitation(BaseModel):
    """Citation shown to users and passed into generated answers."""

    citation_id: int = Field(..., ge=1, description="One-based citation number.")
    chunk_id: str
    company: str
    ticker: str
    fiscal_year: int
    filing_type: FilingType
    section: FilingSectionName
    section_title: str
    source_url: HttpUrl
    excerpt: str = Field(..., min_length=1, description="Short source passage used as evidence.")


class FinSightAnswer(BaseModel):
    """Final grounded answer returned by FinSight."""

    query: str = Field(..., min_length=1, description="Original user question.")
    answer: str = Field(..., description="Grounded answer generated from retrieved SEC context.")
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations supporting the answer.",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Full retrieval trace for debugging and evaluation.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known answer limitations or missing evidence.",
    )
