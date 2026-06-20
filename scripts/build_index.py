import argparse

from backend.chunking.chunker import chunk_filing_sections
from backend.ingestion.filing_fetcher import FilingFetcher
from backend.ingestion.sec_client import SECClient
from backend.parsing.section_extractor import extract_filing_sections
from backend.retrieval.embedding_client import EmbeddingClient
from backend.retrieval.vector_store import LocalVectorStore


def build_index(ticker: str, fiscal_year: int) -> None:
    ticker = ticker.upper()
    index_name = f"{ticker}_{fiscal_year}"

    print(f"Building FinSight index: {index_name}")

    sec_client = SECClient()
    metadata = sec_client.get_10k_metadata(ticker=ticker, fiscal_year=fiscal_year)

    print("Fetched metadata:")
    print(f"  Company: {metadata.company}")
    print(f"  Ticker: {metadata.ticker}")
    print(f"  Fiscal year: {metadata.fiscal_year}")
    print(f"  Filing date: {metadata.filing_date}")
    print(f"  Source: {metadata.source_url}")

    fetcher = FilingFetcher()
    raw_filing = fetcher.fetch_raw_filing(metadata)

    print(f"Raw filing chars: {len(raw_filing.content):,}")

    sections = extract_filing_sections(raw_filing)

    print(f"Extracted sections: {len(sections)}")
    for section in sections:
        print(f"  {section.section}: {len(section.text):,} chars")

    chunks = chunk_filing_sections(sections)

    print(f"Created chunks: {len(chunks)}")
    print(f"Max estimated tokens: {max(chunk.token_count for chunk in chunks)}")

    embedding_client = EmbeddingClient()
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])

    print(f"Created embeddings: {len(embeddings)}")
    print(f"Embedding dim: {len(embeddings[0])}")

    store = LocalVectorStore(index_name=index_name)
    store.save(chunks=chunks, embeddings=embeddings)

    print("Saved local index:")
    print(f"  {store.chunks_path}")
    print(f"  {store.embeddings_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local FinSight index for one 10-K.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. MSFT")
    parser.add_argument("--year", required=True, type=int, help="Fiscal year, e.g. 2023")

    args = parser.parse_args()

    build_index(ticker=args.ticker, fiscal_year=args.year)


if __name__ == "__main__":
    main()