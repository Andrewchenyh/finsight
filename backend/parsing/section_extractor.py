import re

from bs4 import BeautifulSoup

from schemas import RawFiling


def clean_filing_html(raw_filing: RawFiling) -> str:
    """Convert raw SEC filing HTML into normalized plain text."""
    if raw_filing.content_type != "html":
        return normalize_whitespace(raw_filing.content)

    soup = BeautifulSoup(raw_filing.content, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    """Normalize SEC filing whitespace while preserving paragraph boundaries."""
    text = text.replace("\xa0", " ")

    lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r"\s+", " ", line).strip()
        if cleaned_line:
            lines.append(cleaned_line)

    normalized = "\n".join(lines)

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()