from backend.generation.answer_generator import AnswerGenerator
from backend.retrieval.retriever import DenseRetriever
from backend.schemas import FinSightAnswer, FilingSectionName, FilingType, RetrievalFilter
from backend.chunking.chunker import chunk_filing_sections
from backend.ingestion.filing_fetcher import FilingFetcher
from backend.ingestion.sec_client import SECClient
from backend.parsing.section_extractor import extract_filing_sections
from backend.retrieval.embedding_client import EmbeddingClient
from backend.retrieval.vector_store import LocalVectorStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.rerank_retriever import HybridRerankRetriever
from backend.retrieval.retriever import DenseRetriever
from backend.schemas import (
    FinSightAnswer,
    FilingSectionName,
    FilingType,
    RetrievalFilter,
    RetrievalMode,
    RetrievedChunk,
)


def answer_sec_question(
    query: str,
    index_name: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    section: FilingSectionName | None = None,
    filing_type: FilingType | None = None,
    top_k: int = 5,
    retrieval_mode: RetrievalMode = "hybrid_rerank",
) -> FinSightAnswer:
    """Answer a question using a local SEC filing index."""
    retrieved_chunks = retrieve_sec_chunks(
        query=query,
        index_name=index_name,
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
        filing_type=filing_type,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )

    generator = AnswerGenerator()

    return generator.generate_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )
    

def build_sec_index(
    ticker: str,
    fiscal_year: int,
    index_name: str | None = None,
) -> str:
    """Build a local SEC filing index and return the index name."""
    normalized_ticker = ticker.upper().strip()
    resolved_index_name = index_name or f"{normalized_ticker}_{fiscal_year}"

    sec_client = SECClient()
    metadata = sec_client.get_10k_metadata(
        ticker=normalized_ticker,
        fiscal_year=fiscal_year,
    )

    fetcher = FilingFetcher()
    raw_filing = fetcher.fetch_raw_filing(metadata)

    sections = extract_filing_sections(raw_filing)
    chunks = chunk_filing_sections(sections)

    if not chunks:
        raise ValueError(f"No chunks were created for {normalized_ticker} {fiscal_year}.")

    embedding_client = EmbeddingClient()
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])

    store = LocalVectorStore(index_name=resolved_index_name)
    store.save(chunks=chunks, embeddings=embeddings)

    return resolved_index_name


def retrieve_sec_chunks(
    query: str,
    index_name: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    section: FilingSectionName | None = None,
    filing_type: FilingType | None = None,
    top_k: int = 5,
    retrieval_mode: RetrievalMode = "hybrid_rerank",
) -> list[RetrievedChunk]:
    """Retrieve SEC filing chunks using the selected retrieval mode."""
    filters = RetrievalFilter(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
        filing_type=filing_type,
    )

    retriever = _build_retriever(
        index_name=index_name,
        retrieval_mode=retrieval_mode,
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
    )


def _build_retriever(index_name: str, retrieval_mode: RetrievalMode):
    if retrieval_mode == "dense":
        return DenseRetriever(index_name=index_name)

    if retrieval_mode == "bm25":
        return BM25Retriever(index_name=index_name)

    if retrieval_mode == "hybrid":
        return HybridRetriever(index_name=index_name)

    if retrieval_mode == "hybrid_rerank":
        return HybridRerankRetriever(index_name=index_name)

    raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")