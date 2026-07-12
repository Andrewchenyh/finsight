from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CohereReranker
from backend.schemas import RetrievalFilter, RetrievedChunk


DEFAULT_CANDIDATE_K = 15


class HybridRerankRetriever:
    """Hybrid retriever followed by Cohere reranking."""

    def __init__(
        self,
        index_name: str,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CohereReranker | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")

        self.index_name = index_name
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(index_name=index_name)
        self.reranker = reranker or CohereReranker()
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve candidates with hybrid search, then rerank them."""
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        candidate_k = max(top_k, self.candidate_k)

        candidates = self.hybrid_retriever.retrieve(
            query=cleaned_query,
            top_k=candidate_k,
            filters=filters,
        )

        return self.reranker.rerank(
            query=cleaned_query,
            results=candidates,
            top_k=top_k,
        )