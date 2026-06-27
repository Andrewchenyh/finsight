from backend.generation.answer_generator import AnswerGenerator
from backend.retrieval.retriever import DenseRetriever
from backend.schemas import FinSightAnswer, FilingSectionName, FilingType, RetrievalFilter


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