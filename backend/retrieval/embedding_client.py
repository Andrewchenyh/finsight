import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingClient:
    """Small wrapper around OpenAI embedding models."""

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        cleaned_texts = [self._clean_text(text) for text in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned_texts,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_texts([query])[0]

    def _clean_text(self, text: str) -> str:
        """Keep embedding inputs non-empty and reasonably normalized."""
        cleaned = " ".join(text.split())

        if not cleaned:
            raise ValueError("Cannot embed empty text.")

        return cleaned