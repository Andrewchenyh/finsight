from backend.generation.answer_generator import AnswerGenerator
from backend.retrieval.retriever import DenseRetriever
from backend.schemas import FinSightAnswer, FilingSectionName, FilingType, RetrievalFilter
from backend.chunking.chunker import chunk_filing_sections
from backend.ingestion.filing_fetcher import FilingFetcher
from backend.ingestion.sec_client import SECClient
from backend.parsing.section_extractor import extract_filing_sections
from backend.retrieval.embedding_client import EmbeddingClient
from backend.retrieval.vector_store import LocalVectorStore


def answer_sec_question(
    query: str,
    index_name: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    section: FilingSectionName | None = None,
    filing_type: FilingType | None = None,
    top_k: int = 5,
) -> FinSightAnswer:
    """Answer a question using a local SEC filing index."""
    filters = RetrievalFilter(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
        filing_type=filing_type,
    )

    retriever = DenseRetriever(index_name=index_name)
    retrieved_chunks = retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
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