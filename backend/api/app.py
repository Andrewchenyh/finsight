from fastapi import FastAPI

from backend.api.errors import to_http_exception
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from backend.service import answer_sec_question, build_sec_index, retrieve_sec_chunks

app = FastAPI(
    title="FinSight API",
    version="0.1.0",
    description="RAG-powered Q&A over SEC 10-K filings with grounded citations.",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        results = retrieve_sec_chunks(
            query=request.query,
            index_name=request.index_name,
            ticker=request.ticker,
            fiscal_year=request.fiscal_year,
            section=request.section,
            filing_type=request.filing_type,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
        )

        return RetrieveResponse(
            query=request.query,
            index_name=request.index_name,
            retrieval_mode=request.retrieval_mode,
            results=results,
        )

    except Exception as exc:
        raise to_http_exception(exc) from exc
    
    
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = answer_sec_question(
            query=request.query,
            index_name=request.index_name,
            ticker=request.ticker,
            fiscal_year=request.fiscal_year,
            section=request.section,
            filing_type=request.filing_type,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
        )

        return ChatResponse(result=result)

    except Exception as exc:
        raise to_http_exception(exc) from exc
    
    
@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    try:
        index_name = build_sec_index(
            ticker=request.ticker,
            fiscal_year=request.fiscal_year,
            index_name=request.index_name,
        )

        return IngestResponse(
            status="success",
            index_name=index_name,
            message=f"Built local index '{index_name}'.",
        )

    except Exception as exc:
        raise to_http_exception(exc) from exc
