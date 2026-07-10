import os

import cohere
from dotenv import load_dotenv

from backend.schemas import RetrievedChunk


load_dotenv()


DEFAULT_RERANK_MODEL = "rerank-v3.5"


class CohereReranker:
    """Rerank retrieved chunks using Cohere Rerank."""

    def __init__(self, model: str = DEFAULT_RERANK_MODEL):
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError("COHERE_API_KEY is not set.")

        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks by query-document relevance."""
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not results:
            return []

        documents = [self._format_document(result) for result in results]

        response = self.client.rerank(
            model=self.model,
            query=cleaned_query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )

        reranked_results: list[RetrievedChunk] = []

        for rank, item in enumerate(response.results, start=1):
            original_result = results[item.index]

            reranked_results.append(
                RetrievedChunk(
                    chunk=original_result.chunk,
                    score=float(item.relevance_score),
                    rank=rank,
                    retrieval_method="rerank",
                )
            )

        return reranked_results

    def _format_document(self, result: RetrievedChunk) -> str:
        chunk = result.chunk
        metadata = chunk.metadata

        return "\n".join(
            [
                f"Company: {metadata.company}",
                f"Ticker: {metadata.ticker}",
                f"Fiscal year: {metadata.fiscal_year}",
                f"Filing type: {metadata.filing_type}",
                f"Section: {chunk.section} - {chunk.section_title}",
                "",
                chunk.text,
            ]
        )