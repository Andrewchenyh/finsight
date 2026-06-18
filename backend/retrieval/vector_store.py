import json
from pathlib import Path

import numpy as np

from backend.schemas import DocumentChunk, RetrievalFilter, RetrievedChunk


class LocalVectorStore:
    """Simple local vector store backed by JSON chunks and NumPy embeddings."""

    def __init__(
        self,
        index_name: str,
        index_dir: str | Path = "data/index",
    ):
        self.index_name = index_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / f"{self.index_name}_chunks.json"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / f"{self.index_name}_embeddings.npy"

    def save(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Save chunks and embeddings to disk."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})."
            )

        if not chunks:
            raise ValueError("Cannot save an empty index.")

        chunk_payload = [chunk.model_dump(mode="json") for chunk in chunks]
        self.chunks_path.write_text(
            json.dumps(chunk_payload, indent=2),
            encoding="utf-8",
        )

        embedding_array = np.array(embeddings, dtype=np.float32)
        np.save(self.embeddings_path, embedding_array)

    def load(self) -> tuple[list[DocumentChunk], np.ndarray]:
        """Load chunks and embeddings from disk."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks file: {self.chunks_path}")

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {self.embeddings_path}")

        chunk_payload = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk.model_validate(item) for item in chunk_payload]
        embeddings = np.load(self.embeddings_path)

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding rows ({embeddings.shape[0]})."
            )

        return chunks, embeddings

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Search the local index with dot-product similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        chunks, embeddings = self.load()

        query_vector = np.array(query_embedding, dtype=np.float32)

        if query_vector.ndim != 1:
            raise ValueError("query_embedding must be one-dimensional.")

        if embeddings.shape[1] != query_vector.shape[0]:
            raise ValueError(
                f"Query dim ({query_vector.shape[0]}) does not match index dim ({embeddings.shape[1]})."
            )

        candidate_indices = self._filter_indices(chunks=chunks, filters=filters)

        if not candidate_indices:
            return []

        candidate_embeddings = embeddings[candidate_indices]
        scores = candidate_embeddings @ query_vector

        ranked_local_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []

        for rank, local_index in enumerate(ranked_local_indices, start=1):
            global_index = candidate_indices[int(local_index)]
            score = float(scores[int(local_index)])

            results.append(
                RetrievedChunk(
                    chunk=chunks[global_index],
                    score=score,
                    rank=rank,
                    retrieval_method="dense",
                )
            )

        return results

    def _filter_indices(
        self,
        chunks: list[DocumentChunk],
        filters: RetrievalFilter | None,
    ) -> list[int]:
        """Return chunk indices matching optional metadata filters."""
        if filters is None:
            return list(range(len(chunks)))

        matching_indices: list[int] = []

        for index, chunk in enumerate(chunks):
            if filters.ticker is not None and chunk.metadata.ticker != filters.ticker:
                continue

            if filters.fiscal_year is not None and chunk.metadata.fiscal_year != filters.fiscal_year:
                continue

            if filters.section is not None and chunk.section != filters.section:
                continue

            if filters.filing_type is not None and chunk.metadata.filing_type != filters.filing_type:
                continue

            matching_indices.append(index)

        return matching_indices