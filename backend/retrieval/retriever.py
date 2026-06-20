from backend.retrieval.embedding_client import EmbeddingClient
from backend.retrieval.vector_store import LocalVectorStore
from backend.schemas import RetrievalFilter, RetrievedChunk


class DenseRetriever:
    """Dense retriever backed by OpenAI embeddings and LocalVectorStore."""

    def __init__(
        self,
        index_name: str,
        embedding_client: EmbeddingClient | None = None,
        vector_store: LocalVectorStore | None = None,
    ):
        self.index_name = index_name
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_store = vector_store or LocalVectorStore(index_name=index_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a natural-language query."""
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        query_embedding = self.embedding_client.embed_query(cleaned_query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )