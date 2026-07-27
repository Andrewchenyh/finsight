from pydantic import BaseModel, Field

from backend.schemas import FilingSectionName, RetrievalMode


class EvalQuestion(BaseModel):
    """One retrieval evaluation question."""

    id: str = Field(..., min_length=1, description="Stable eval question ID.")
    query: str = Field(..., min_length=1, description="Natural-language question.")
    ticker: str = Field(..., min_length=1, description="Ticker symbol.")
    fiscal_year: int = Field(..., ge=1994, description="Fiscal year.")
    section: FilingSectionName | None = Field(
        default=None,
        description="Optional filing section filter.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks.")
    expected_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs considered correct.",
    )
    expected_terms: list[str] = Field(
        default_factory=list,
        description="Terms or phrases expected to appear in relevant chunks.",
    )
    notes: str | None = Field(
        default=None,
        description="Human notes about what the query is testing.",
    )


class EvalDataset(BaseModel):
    """A retrieval evaluation dataset."""

    name: str = Field(..., min_length=1)
    index_name: str = Field(..., min_length=1)
    questions: list[EvalQuestion] = Field(..., min_length=1)


class EvalQuestionResult(BaseModel):
    """Metrics for one query under one retrieval mode."""

    question_id: str
    retrieval_mode: RetrievalMode
    retrieved_chunk_ids: list[str]
    expected_chunk_hit: bool
    expected_chunk_rank: int | None
    expected_terms_hit_count: int
    expected_terms_total: int
    precision_at_k: float


class EvalRunResult(BaseModel):
    """Aggregate metrics for one retrieval mode."""

    dataset_name: str
    retrieval_mode: RetrievalMode
    question_results: list[EvalQuestionResult]
    recall_at_k: float
    mean_precision_at_k: float
    mean_expected_chunk_rank: float | None