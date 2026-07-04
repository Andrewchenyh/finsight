import re

import numpy as np
from rank_bm25 import BM25Okapi

from backend.retrieval.vector_store import LocalVectorStore
from backend.schemas import DocumentChunk, RetrievalFilter, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


class BM25Retriever:
    """Lexical retriever backed by BM25 over locally stored chunks."""

    def __init__(
        self,
        index_name: str,
        vector_store: LocalVectorStore | None = None,
    ):
        self.index_name = index_name
        self.vector_store = vector_store or LocalVectorStore(index_name=index_name)

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

        chunks, _ = self.vector_store.load()
        candidate_chunks = self._filter_chunks(chunks=chunks, filters=filters)

        if not candidate_chunks:
            return []

        tokenized_corpus = [tokenize(chunk.text) for chunk in candidate_chunks]
        tokenized_query = tokenize(cleaned_query)

        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float32)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []

        for rank, candidate_index in enumerate(ranked_indices, start=1):
            score = float(scores[int(candidate_index)])

            if score <= 0:
                continue

            results.append(
                RetrievedChunk(
                    chunk=candidate_chunks[int(candidate_index)],
                    score=score,
                    rank=rank,
                    retrieval_method="bm25",
                )
            )

        return results

    def _filter_chunks(
        self,
        chunks: list[DocumentChunk],
        filters: RetrievalFilter | None,
    ) -> list[DocumentChunk]:
        if filters is None:
            return chunks

        matching_chunks: list[DocumentChunk] = []

        for chunk in chunks:
            if filters.ticker is not None and chunk.metadata.ticker != filters.ticker:
                continue

            if filters.fiscal_year is not None and chunk.metadata.fiscal_year != filters.fiscal_year:
                continue

            if filters.section is not None and chunk.section != filters.section:
                continue

            if filters.filing_type is not None and chunk.metadata.filing_type != filters.filing_type:
                continue

            matching_chunks.append(chunk)

        return matching_chunks
