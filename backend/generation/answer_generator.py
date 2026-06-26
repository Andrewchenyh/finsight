import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.schemas import FinSightAnswer, RetrievedChunk, SourceCitation


load_dotenv()


DEFAULT_GENERATION_MODEL = "gpt-4o-mini"


class AnswerGenerator:
    """Generate SEC-grounded answers from retrieved filing chunks."""

    def __init__(self, model: str = DEFAULT_GENERATION_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> FinSightAnswer:
        """Generate a grounded answer with source citations."""
        cleaned_query = " ".join(query.split())

        if not cleaned_query:
            raise ValueError("query must not be empty.")

        if not retrieved_chunks:
            return FinSightAnswer(
                query=cleaned_query,
                answer="I could not find enough relevant SEC filing context to answer this question.",
                citations=[],
                retrieved_chunks=[],
                limitations=["No relevant chunks were retrieved."],
            )

        citations = self._build_citations(retrieved_chunks)
        context = self._build_context(citations)

        prompt = self._build_prompt(
            query=cleaned_query,
            context=context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are FinSight, an SEC filing research assistant. "
                        "Answer only using the provided SEC filing excerpts. "
                        "Do not use outside knowledge. "
                        "If the excerpts do not contain enough evidence, say so. "
                        "Cite claims with bracketed citation IDs like [1] or [2]."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content or ""

        return FinSightAnswer(
            query=cleaned_query,
            answer=answer.strip(),
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            limitations=[],
        )

    def _build_citations(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[SourceCitation]:
        citations: list[SourceCitation] = []

        for index, retrieved_chunk in enumerate(retrieved_chunks, start=1):
            chunk = retrieved_chunk.chunk
            metadata = chunk.metadata

            citations.append(
                SourceCitation(
                    citation_id=index,
                    chunk_id=chunk.chunk_id,
                    company=metadata.company,
                    ticker=metadata.ticker,
                    fiscal_year=metadata.fiscal_year,
                    filing_type=metadata.filing_type,
                    section=chunk.section,
                    section_title=chunk.section_title,
                    source_url=metadata.source_url,
                    excerpt=self._shorten_excerpt(chunk.text),
                )
            )

        return citations

    def _build_context(self, citations: list[SourceCitation]) -> str:
        context_blocks = []

        for citation in citations:
            context_blocks.append(
                "\n".join(
                    [
                        f"[{citation.citation_id}] "
                        f"{citation.company} {citation.fiscal_year} {citation.filing_type}, "
                        f"{citation.section} - {citation.section_title}",
                        citation.excerpt,
                    ]
                )
            )

        return "\n\n".join(context_blocks)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
                    Question:
                    {query}

                    SEC filing excerpts:
                    {context}

                    Instructions:
                    - Answer the question using only the excerpts above.
                    - Cite every material claim with citation IDs like [1].
                    - If the excerpts are incomplete, mention the limitation.
                    - Be concise but specific.
                """.strip()

    def _shorten_excerpt(self, text: str, max_chars: int = 1_200) -> str:
        cleaned = " ".join(text.split())

        if len(cleaned) <= max_chars:
            return cleaned

        return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."