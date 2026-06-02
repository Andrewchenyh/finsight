import os
from typing import Any

import requests
from dotenv import load_dotenv

from backend.schemas import FilingMetadata


load_dotenv()


SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


class SECClient:
    """Small client for SEC ticker lookup and filing metadata retrieval."""

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")

        if not self.user_agent:
            raise ValueError(
                "SEC_USER_AGENT is not set. Use a value like "
                "'AI Investment Copilot andrew@example.com'."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def get_ticker_map(self) -> dict[str, dict[str, Any]]:
        """Return SEC ticker metadata keyed by uppercased ticker."""
        response = self.session.get(SEC_TICKER_URL, timeout=20)
        response.raise_for_status()

        raw_mapping = response.json()

        return {
            company["ticker"].upper(): company
            for company in raw_mapping.values()
        }

    def get_cik_for_ticker(self, ticker: str) -> str:
        """Resolve a stock ticker to an unpadded SEC CIK string."""
        ticker = ticker.upper().strip()
        ticker_map = self.get_ticker_map()

        if ticker not in ticker_map:
            raise ValueError(f"Ticker '{ticker}' was not found in SEC ticker mapping.")

        return str(ticker_map[ticker]["cik_str"])

    def get_company_name_for_ticker(self, ticker: str) -> str:
        """Resolve a stock ticker to the SEC company title."""
        ticker = ticker.upper().strip()
        ticker_map = self.get_ticker_map()

        if ticker not in ticker_map:
            raise ValueError(f"Ticker '{ticker}' was not found in SEC ticker mapping.")

        return ticker_map[ticker]["title"]

    def get_submissions(self, cik: str) -> dict[str, Any]:
        """Fetch SEC submissions metadata for one CIK."""
        cik_padded = cik.zfill(10)
        url = SEC_SUBMISSIONS_URL.format(cik_padded=cik_padded)

        response = self.session.get(url, timeout=20)
        response.raise_for_status()

        return response.json()

    def get_10k_metadata(self, ticker: str, fiscal_year: int) -> FilingMetadata:
        """Find the 10-K metadata for a ticker and fiscal year."""
        ticker = ticker.upper().strip()
        cik = self.get_cik_for_ticker(ticker)
        company = self.get_company_name_for_ticker(ticker)
        submissions = self.get_submissions(cik)

        recent = submissions["filings"]["recent"]

        for index, form in enumerate(recent["form"]):
            if form != "10-K":
                continue

            report_date = recent["reportDate"][index]
            filing_date = recent["filingDate"][index]
            accession_number = recent["accessionNumber"][index]
            primary_document = recent["primaryDocument"][index]

            if not report_date.startswith(str(fiscal_year)):
                continue

            accession_no_dashes = accession_number.replace("-", "")
            source_url = (
                f"{SEC_ARCHIVES_BASE_URL}/"
                f"{cik}/"
                f"{accession_no_dashes}/"
                f"{primary_document}"
            )

            return FilingMetadata(
                company=company,
                ticker=ticker,
                cik=cik,
                accession_number=accession_number,
                filing_type="10-K",
                fiscal_year=fiscal_year,
                filing_date=filing_date,
                source_url=source_url,
            )

        raise ValueError(f"No 10-K found for {ticker} fiscal year {fiscal_year}.")