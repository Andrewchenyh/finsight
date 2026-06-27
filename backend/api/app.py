from fastapi import FastAPI, HTTPException

from backend.api.schemas import HealthResponse, RetrieveRequest, RetrieveResponse
from backend.retrieval.retriever import DenseRetriever


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