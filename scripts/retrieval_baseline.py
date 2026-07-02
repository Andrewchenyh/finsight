from backend.retrieval.retriever import DenseRetriever
from backend.schemas import RetrievalFilter


BASELINE_QUERIES = [
    {
        "name": "cybersecurity_risks",
        "query": "What cybersecurity risks does Microsoft describe?",
        "section": "Item 1A",
    },
    {
        "name": "ai_risks",
        "query": "What risks does Microsoft describe related to AI?",
        "section": "Item 1A",
    },
    {
        "name": "competition",
        "query": "How does Microsoft describe competition in its business?",
        "section": "Item 1",
    },
    {
        "name": "cloud_infrastructure",
        "query": "What does Microsoft say about cloud infrastructure risks?",
        "section": "Item 1A",
    },
    {
        "name": "economic_conditions_mda",
        "query": "What does Microsoft say in MD&A about economic conditions?",
        "section": "Item 7",
    },
]


def print_results() -> None:
    retriever = DenseRetriever(index_name="MSFT_2023")

    for item in BASELINE_QUERIES:
        query = item["query"]
        section = item["section"]

        results = retriever.retrieve(
            query=query,
            top_k=5,
            filters=RetrievalFilter(
                ticker="MSFT",
                fiscal_year=2023,
                section=section,  # type: ignore[arg-type]
            ),
        )

        print("=" * 100)
        print(f"CASE: {item['name']}")
        print(f"QUERY: {query}")
        print(f"FILTER SECTION: {section}")
        print()

        for result in results:
            chunk = result.chunk
            preview = " ".join(chunk.text.split())[:350]

            print(
                f"Rank {result.rank} | "
                f"Score {result.score:.4f} | "
                f"{chunk.section} | "
                f"{chunk.chunk_id}"
            )
            print(preview)
            print()


if __name__ == "__main__":
    print_results()