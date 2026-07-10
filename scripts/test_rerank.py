from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CohereReranker
from backend.schemas import RetrievalFilter


QUERIES = [
    ("cybersecurity", "What cybersecurity risks does Microsoft describe?", "Item 1A"),
    ("competition", "How does Microsoft describe competition in its business?", "Item 1"),
    ("economic_conditions", "What does Microsoft say in MD&A about economic conditions?", "Item 7"),
]


def main() -> None:
    retriever = HybridRetriever(index_name="MSFT_2023")
    reranker = CohereReranker()

    for name, query, section in QUERIES:
        print("=" * 100)
        print(name)

        candidates = retriever.retrieve(
            query=query,
            top_k=15,
            filters=RetrievalFilter(
                ticker="MSFT",
                fiscal_year=2023,
                section=section,  # type: ignore[arg-type]
            ),
        )

        reranked = reranker.rerank(
            query=query,
            results=candidates,
            top_k=5,
        )

        for result in reranked:
            preview = " ".join(result.chunk.text.split())[:500]
            print(
                f"Rank {result.rank} | "
                f"Score {result.score:.4f} | "
                f"{result.chunk.section} | "
                f"{result.chunk.chunk_id}"
            )
            print(preview)
            print()


if __name__ == "__main__":
    main()