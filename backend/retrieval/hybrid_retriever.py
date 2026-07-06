from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.retriever import DenseRetriever
from backend.schemas import RetrievalFilter, RetrievedChunk


DEFAULT_RRF_K = 60
DEFAULT_FETCH_K = 20


class HybridRetriever:
    """Hybrid dense + BM25 retriever fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        index_name: str,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        fetch_k: int = DEFAULT_FETCH_K,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        if fetch_k <= 0:
            raise ValueError("fetch_k must be positive.")

        self.index_name = index_name
        self.dense_retriever = dense_retriever or DenseRetriever(index_name=index_name)
        self.bm25_retriever = bm25_retriever or BM25Retriever(index_name=index_name)
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using dense + BM25 results fused by RRF."""
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        fetch_k = max(top_k, self.fetch_k)

        dense_results = self.dense_retriever.retrieve(
            query=cleaned_query,
            top_k=fetch_k,
            filters=filters,
        )
        bm25_results = self.bm25_retriever.retrieve(
            query=cleaned_query,
            top_k=fetch_k,
            filters=filters,
        )

        return reciprocal_rank_fusion(
            result_lists=[dense_results, bm25_results],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse ranked retrieval lists using Reciprocal Rank Fusion."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    fused_scores: dict[str, float] = {}
    best_chunks: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + result.rank
            )

            if chunk_id not in best_chunks:
                best_chunks[chunk_id] = result

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    fused_results: list[RetrievedChunk] = []

    for rank, chunk_id in enumerate(ranked_chunk_ids[:top_k], start=1):
        original_result = best_chunks[chunk_id]

        fused_results.append(
            RetrievedChunk(
                chunk=original_result.chunk,
                score=fused_scores[chunk_id],
                rank=rank,
                retrieval_method="hybrid",
            )
        )

    return fused_results