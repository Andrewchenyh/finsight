from fastapi import FastAPI, HTTPException

from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from backend.retrieval.retriever import DenseRetriever
from backend.service import answer_sec_question, build_sec_index

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
        retriever = DenseRetriever(index_name=request.index_name)
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.to_filter(),
        )

        return RetrieveResponse(
            query=request.query,
            index_name=request.index_name,
            results=results,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    
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
        )

        return ChatResponse(result=result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    
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

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc