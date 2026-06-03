import re

from bs4 import BeautifulSoup

from backend.schemas import FilingSection, FilingSectionName, RawFiling


SECTION_TITLES: dict[str, str] = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 1B": "Unresolved Staff Comments",
    "Item 2": "Properties",
    "Item 3": "Legal Proceedings",
    "Item 4": "Mine Safety Disclosures",
    "Item 5": "Market for Registrant's Common Equity",
    "Item 6": "Reserved",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements and Supplementary Data",
    "Item 9": "Changes in and Disagreements with Accountants",
    "Item 9A": "Controls and Procedures",
    "Item 9B": "Other Information",
    "Item 9C": "Disclosure Regarding Foreign Jurisdictions",
    "Item 10": "Directors, Executive Officers and Corporate Governance",
    "Item 11": "Executive Compensation",
    "Item 12": "Security Ownership",
    "Item 13": "Certain Relationships and Related Transactions",
    "Item 14": "Principal Accountant Fees and Services",
    "Item 15": "Exhibits and Financial Statement Schedules",
}

TARGET_SECTIONS: tuple[str, ...] = (
    "Item 1",
    "Item 1A",
    "Item 7",
    "Item 7A",
    "Item 8",
)

ITEM_HEADING_PATTERN = re.compile(
    r"""
    ^
    \s*
    item
    \s+
    (?P<number>
        1A|1B|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15
    )
    \.?
    (?:
        \s+
        (?P<title>[^\n]{0,160})
    )?
    \s*
    $
    """,
    flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


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


def extract_filing_sections(
    raw_filing: RawFiling,
    target_sections: tuple[str, ...] = TARGET_SECTIONS,
    min_section_chars: int = 1_000,
) -> list[FilingSection]:
    """Extract major 10-K sections from a raw filing.

    SEC filings often repeat Item headings in a table of contents. To avoid
    selecting tiny table-of-contents spans, this function keeps the largest
    plausible span found for each target section.
    """
    clean_text = clean_filing_html(raw_filing)
    heading_matches = _find_item_headings(clean_text)

    if len(heading_matches) < 2:
        raise ValueError("Could not find enough 10-K Item headings to extract sections.")

    best_spans: dict[str, tuple[int, int]] = {}

    for index, heading in enumerate(heading_matches[:-1]):
        section_name = heading["section"]

        if section_name not in target_sections:
            continue

        start = heading["start"]
        end = heading_matches[index + 1]["start"]

        if end <= start:
            continue

        section_length = end - start

        if section_length < min_section_chars:
            continue

        existing_span = best_spans.get(section_name)
        if existing_span is None or section_length > (existing_span[1] - existing_span[0]):
            best_spans[section_name] = (start, end)

    sections: list[FilingSection] = []

    for section_name in target_sections:
        span = best_spans.get(section_name)

        if span is None:
            continue

        start, end = span
        section_text = clean_text[start:end].strip()

        sections.append(
            FilingSection(
                metadata=raw_filing.metadata,
                section=section_name,  # type: ignore[arg-type]
                section_title=SECTION_TITLES.get(section_name, "Unknown"),
                text=section_text,
                char_start=start,
                char_end=end,
            )
        )

    return sections


def _find_item_headings(clean_text: str) -> list[dict[str, int | str]]:
    """Find candidate 10-K Item headings in cleaned filing text."""
    headings: list[dict[str, int | str]] = []

    for match in ITEM_HEADING_PATTERN.finditer(clean_text):
        item_number = match.group("number").upper()
        section_name = f"Item {item_number}"

        if section_name not in SECTION_TITLES:
            continue

        headings.append(
            {
                "section": section_name,
                "start": match.start(),
                "end": match.end(),
            }
        )

    return headings