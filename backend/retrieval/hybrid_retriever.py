from concurrent.futures import ThreadPoolExecutor

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.retriever import DenseRetriever
from backend.schemas import RetrievalFilter, RetrievedChunk


DEFAULT_RRF_K = 10
DEFAULT_FETCH_K = 20


class HybridRetriever:
    """Hybrid dense + BM25 retriever fused with Reciprocal Rank Fusion.

    Dense and BM25 retrieval are run concurrently in a thread pool. The dense
    retriever is I/O-bound (OpenAI API call), so it releases Python's GIL and
    allows the BM25 CPU work to overlap. Total latency becomes
    max(dense_time, bm25_time) rather than dense_time + bm25_time.
    """

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
        """Retrieve chunks using dense + BM25 results fused by RRF.

        Each sub-retriever fetches fetch_k >= top_k candidates so that RRF
        has enough overlap to work with. The final list is trimmed to top_k
        after fusion.
        """
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        fetch_k = max(top_k, self.fetch_k)

        # Submit both retrievals concurrently. The dense retriever makes a
        # network call to the OpenAI embeddings API; the BM25 retriever does
        # CPU work over the in-memory index. Because the network call releases
        # the GIL, both can make progress at the same time.
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.dense_retriever.retrieve,
                cleaned_query,
                fetch_k,
                filters,
            )
            bm25_future = executor.submit(
                self.bm25_retriever.retrieve,
                cleaned_query,
                fetch_k,
                filters,
            )
            dense_results = dense_future.result()
            bm25_results = bm25_future.result()

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
    """Fuse ranked retrieval lists using Reciprocal Rank Fusion.

    RRF score for a chunk = sum over all lists of 1 / (rrf_k + rank).

    Key properties:
    - Scale-invariant: raw scores from each retriever are discarded; only
      rank position matters. This makes it safe to fuse dense cosine scores
      (range ~0–1) with BM25 TF-IDF scores (unbounded range).
    - Robust to rank outliers: rrf_k=60 (Cormack et al., 2009) prevents a
      single top-ranked result from dominating the fused score.
    """
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