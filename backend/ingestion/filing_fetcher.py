import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from backend.schemas import FilingMetadata, RawFiling


load_dotenv()


class FilingFetcher:
    """Downloads raw SEC filing documents and optionally caches them locally."""

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: str | Path = "data/sec_filings/raw",
    ):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")

        if not self.user_agent:
            raise ValueError(
                "SEC_USER_AGENT is not set. Use a value like "
                "'AI Investment Copilot your_email@example.com'."
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _cache_path(self, metadata: FilingMetadata) -> Path:
        safe_accession = metadata.accession_number.replace("-", "")
        filename = f"{metadata.ticker}_{metadata.fiscal_year}_{safe_accession}.html"
        return self.cache_dir / filename

    def fetch_raw_filing(
        self,
        metadata: FilingMetadata,
        use_cache: bool = True,
    ) -> RawFiling:
        """Fetch raw filing HTML from SEC and return a RawFiling object."""
        cache_path = self._cache_path(metadata)

        if use_cache and cache_path.exists():
            content = cache_path.read_text(encoding="utf-8")
            return RawFiling(
                metadata=metadata,
                content=content,
                content_type="html",
            )

        response = self.session.get(str(metadata.source_url), timeout=30)
        response.raise_for_status()

        content = response.text

        if not content.strip():
            raise ValueError(f"Downloaded filing is empty: {metadata.source_url}")

        if use_cache:
            cache_path.write_text(content, encoding="utf-8")

        return RawFiling(
            metadata=metadata,
            content=content,
            content_type="html",
        )