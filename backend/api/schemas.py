from pydantic import BaseModel, Field

from backend.schemas import (
    FilingSectionName,
    FilingType,
    FinSightAnswer,
    RetrievedChunk,
    RetrievalFilter,
)


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status.")
    service: str = Field(default="finsight", description="Service name.")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language retrieval query.")
    index_name: str = Field(..., min_length=1, description="Local index name, e.g. MSFT_2023.")
    ticker: str | None = Field(default=None, description="Optional ticker filter.")
    fiscal_year: int | None = Field(default=None, ge=1994, description="Optional fiscal year filter.")
    section: FilingSectionName | None = Field(default=None, description="Optional 10-K section filter.")
    filing_type: FilingType | None = Field(default=None, description="Optional filing type filter.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")

    def to_filter(self) -> RetrievalFilter:
        return RetrievalFilter(
            ticker=self.ticker,
            fiscal_year=self.fiscal_year,
            section=self.section,
            filing_type=self.filing_type,
        )


class RetrieveResponse(BaseModel):
    query: str
    index_name: str
    results: list[RetrievedChunk]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language SEC filing question.")
    index_name: str = Field(..., min_length=1, description="Local index name, e.g. MSFT_2023.")
    ticker: str | None = Field(default=None, description="Optional ticker filter.")
    fiscal_year: int | None = Field(default=None, ge=1994, description="Optional fiscal year filter.")
    section: FilingSectionName | None = Field(default=None, description="Optional 10-K section filter.")
    filing_type: FilingType | None = Field(default=None, description="Optional filing type filter.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")

    def to_filter(self) -> RetrievalFilter:
        return RetrievalFilter(
            ticker=self.ticker,
            fiscal_year=self.fiscal_year,
            section=self.section,
            filing_type=self.filing_type,
        )


class ChatResponse(BaseModel):
    result: FinSightAnswer


class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker symbol to ingest.")
    fiscal_year: int = Field(..., ge=1994, description="Fiscal year to ingest.")
    index_name: str | None = Field(
        default=None,
        description="Optional custom index name. Defaults to TICKER_YEAR.",
    )


class IngestResponse(BaseModel):
    status: str
    index_name: str
    message: str