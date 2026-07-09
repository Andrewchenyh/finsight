import re

import numpy as np
from rank_bm25 import BM25Okapi

from backend.retrieval.vector_store import LocalVectorStore
from backend.schemas import DocumentChunk, RetrievalFilter, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


class BM25Retriever:
    """Lexical retriever backed by BM25 over locally stored chunks.

    The BM25 index is built lazily on the first call to retrieve() and then
    reused for all subsequent calls. This means the O(N) index construction
    cost is paid once regardless of how many queries are run.

    Tradeoff: the index is built over the full corpus, so BM25 IDF statistics
    reflect all chunks (across all tickers, years, sections). Metadata filters
    are applied to the ranked results rather than to the corpus before scoring.
    In practice this is fine for 10-K retrieval — the IDF difference is minor
    compared to the latency savings from building the index once.
    """

    def __init__(
        self,
        index_name: str,
        vector_store: LocalVectorStore | None = None,
    ):
        self.index_name = index_name
        self.vector_store = vector_store or LocalVectorStore(index_name=index_name)

        # Populated on first retrieve() call via _ensure_index().
        self._chunks: list[DocumentChunk] | None = None
        self._bm25: BM25Okapi | None = None

    def _ensure_index(self) -> None:
        """Build BM25 index over the full corpus, lazily on first call."""
        if self._bm25 is not None:
            return  # already built; nothing to do

        self._chunks, _ = self.vector_store.load()

        if not self._chunks:
            return  # empty corpus; _bm25 stays None

        tokenized_corpus = [tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks using BM25 keyword scoring."""
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self._ensure_index()

        if self._bm25 is None:  # empty corpus after load
            return []

        tokenized_query = tokenize(cleaned_query)

        if not tokenized_query:
            return []

        scores = np.array(self._bm25.get_scores(tokenized_query), dtype=np.float32)

        # Descending sort: highest BM25 score first.
        ranked_indices = np.argsort(scores)[::-1]

        results: list[RetrievedChunk] = []
        rank = 0  # incremented only when a chunk is actually kept

        for candidate_index in ranked_indices:
            score = float(scores[int(candidate_index)])

            if score <= 0:
                # Scores are sorted descending; every remaining score is also
                # <= 0. Stop early rather than iterating the full corpus.
                break

            chunk = self._chunks[int(candidate_index)]

            if not self._matches_filters(chunk, filters):
                # Skip this chunk but do NOT increment rank — that was the
                # original bug. Rank must reflect position in the kept list,
                # not position in the full scored list.
                continue

            rank += 1
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="bm25",
                )
            )

            if rank >= top_k:
                break

        return results

    def _matches_filters(
        self,
        chunk: DocumentChunk,
        filters: RetrievalFilter | None,
    ) -> bool:
        """Return True if the chunk satisfies all active filter criteria."""
        if filters is None:
            return True

        if filters.ticker is not None and chunk.metadata.ticker != filters.ticker:
            return False

        if filters.fiscal_year is not None and chunk.metadata.fiscal_year != filters.fiscal_year:
            return False

        if filters.section is not None and chunk.section != filters.section:
            return False

        if filters.filing_type is not None and chunk.metadata.filing_type != filters.filing_type:
            return False

        return True


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 keyword retrieval."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]