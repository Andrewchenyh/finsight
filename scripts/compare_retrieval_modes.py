from backend.service import retrieve_sec_chunks
from backend.schemas import RetrievalMode


QUERY_CASES = [
    {
        "name": "cybersecurity",
        "query": "What cybersecurity risks does Microsoft describe?",
        "section": "Item 1A",
        "expected_chunk_hint": "Security of our products, services, devices, and customers’ data",
    },
    {
        "name": "competition",
        "query": "How does Microsoft describe competition in its business?",
        "section": "Item 1",
        "expected_chunk_hint": "Competition",
    },
    {
        "name": "economic_conditions",
        "query": "What does Microsoft say in MD&A about economic conditions?",
        "section": "Item 7",
        "expected_chunk_hint": "Economic Conditions, Challenges, and Risks",
    },
]

RETRIEVAL_MODES: list[RetrievalMode] = [
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
]


def main() -> None:
    for case in QUERY_CASES:
        print("=" * 100)
        print(f"CASE: {case['name']}")
        print(f"QUERY: {case['query']}")
        print(f"EXPECTED HINT: {case['expected_chunk_hint']}")
        print()

        for mode in RETRIEVAL_MODES:
            results = retrieve_sec_chunks(
                query=case["query"],
                index_name="MSFT_2023",
                ticker="MSFT",
                fiscal_year=2023,
                section=case["section"],  # type: ignore[arg-type]
                top_k=5,
                retrieval_mode=mode,
            )

            print(f"MODE: {mode}")

            if not results:
                print("  No results")
                print()
                continue

            top = results[0]
            preview = " ".join(top.chunk.text.split())[:300]
            hint_found_rank = find_hint_rank(
                results=results,
                hint=case["expected_chunk_hint"],
            )

            print(f"  Top chunk: {top.chunk.chunk_id}")
            print(f"  Top score: {top.score:.4f}")
            print(f"  Top preview: {preview}")
            print(f"  Expected hint rank: {hint_found_rank or 'not found in top 5'}")
            print()


def find_hint_rank(results, hint: str) -> int | None:
    normalized_hint = hint.lower()

    for result in results:
        if normalized_hint in result.chunk.text.lower():
            return result.rank

    return None


if __name__ == "__main__":
    main()