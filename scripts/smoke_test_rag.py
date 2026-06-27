import argparse

from backend.service import answer_sec_question


def run_smoke_test(
    index_name: str,
    ticker: str,
    fiscal_year: int,
    section: str | None,
    query: str,
) -> None:
    answer = answer_sec_question(
        query=query,
        index_name=index_name,
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,  # type: ignore[arg-type]
        top_k=5,
    )

    print("QUESTION")
    print(answer.query)
    print()

    print("ANSWER")
    print(answer.answer)
    print()

    print("CITATIONS")
    for citation in answer.citations:
        print(
            f"[{citation.citation_id}] "
            f"{citation.ticker} {citation.fiscal_year} "
            f"{citation.filing_type} "
            f"{citation.section} - {citation.section_title}"
        )
        print(f"Chunk: {citation.chunk_id}")
        print(citation.excerpt[:400])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FinSight local RAG smoke test.")
    parser.add_argument("--index", default="MSFT_2023", help="Local index name.")
    parser.add_argument("--ticker", default="MSFT", help="Ticker symbol.")
    parser.add_argument("--year", default=2023, type=int, help="Fiscal year.")
    parser.add_argument("--section", default="Item 1A", help="Optional 10-K section filter.")
    parser.add_argument(
        "--query",
        default="What cybersecurity risks does Microsoft describe?",
        help="Question to ask against the local index.",
    )

    args = parser.parse_args()

    section = args.section if args.section else None

    run_smoke_test(
        index_name=args.index,
        ticker=args.ticker,
        fiscal_year=args.year,
        section=section,
        query=args.query,
    )


if __name__ == "__main__":
    main()