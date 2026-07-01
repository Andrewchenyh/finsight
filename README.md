# FinSight

> A RAG-powered **SEC 10-K research assistant**. Ask grounded questions over real company filings, compare risk disclosures across companies and years, and verify every answer with citations traced back to the exact filing section.

FinSight is built as a standalone system — demoable and deployable on its own. It is also designed so that it can later plug into a broader [AI Investment Copilot](https://github.com/Andrewchenyh/ai-investment-copilot) as a dedicated filing research service.

---

## What it does

Ask a question like *"What risks does Microsoft disclose related to cybersecurity in their 2023 10-K?"* and FinSight:

1. Retrieves the most relevant chunks from the indexed filing
2. Generates a grounded answer with inline citations
3. Explicitly flags when retrieved evidence is insufficient to support a claim

```
Answer:
Microsoft's 2023 10-K identifies cyberattacks, nation-state threats, and
vulnerabilities in third-party software as material cybersecurity risks...

Evidence:
[1] MSFT 2023 10-K · Item 1A · Risk Factors
[2] MSFT 2023 10-K · Item 7 · MD&A

Limitations:
Figures above reflect management disclosures, not independently audited
incident data.
```

The full pipeline — fetch, parse, chunk, embed, retrieve, generate — now runs end to end locally against a real filing (MSFT 2023), verified via `scripts/smoke_test_rag.py`.

---

## Architecture

```
   CLI (scripts/)   /   FastAPI (backend/api/app.py)   /   Streamlit (frontend/apps/streamlit_app.py)
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                  FinSight Service (backend/service.py)     │
│   answer_sec_question(query, ticker, year, section)        │
│   retrieve_sec_chunks(query, filters)                      │
│   ingest_10k(ticker, year)                                 │
│   compare_filings(query, tickers, years)                   │
└──────────┬────────────────────────────────────────┬────────┘
           │                                        │
           ▼                                        ▼
┌───────────────────────────────────┐      ┌───────────────────────────────────┐
│  Ingestion → Parsing → Chunking   │      │  Retrieval → Generation           │
│                                   │      │                                   │
│  backend/ingestion/               │      │  backend/retrieval/               │
│    sec_client.py                  │      │    embedding_client.py            │
│      → CIK lookup, EDGAR fetch    │      │      → OpenAI embeddings          │
│    filing_fetcher.py              │      │    vector_store.py                │
│      → HTML download + cache      │      │      → local JSON + NumPy index   │
│                                   │      │    retriever.py                   │
│  backend/parsing/                 │      │      → cosine similarity search   │
│    section_extractor.py           │      │                                   │
│      → Item 1/1A/7/7A/8 extract   │      │  backend/generation/              │
│                                   │      │    answer_generator.py            │
│  backend/chunking/                │      │      → grounded answer + cites    │
│    chunker.py                     │      │      → uncertainty flagging       │
│      → section-aware, token-      │      │                                   │
│        budgeted, stable chunk IDs │      │                                   │
└───────────────────────────────────┘      └───────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  Local Index  (data/index/)          →  Pinecone (Phase 7+) │
│    MSFT_2023_chunks.json                                    │
│    MSFT_2023_embeddings.npy                                 │
│    test_msft_2023_chunks.json        (test fixtures)        │
│    test_msft_2023_embeddings.npy                            │
│                                                             │
│  Rich metadata per chunk:                                   │
│  ticker · cik · accession_number · year                     │
│  section · section_title · chunk_type                       │
│  source_url · char_start · char_end · token_count           │
└─────────────────────────────────────────────────────────────┘
```

---

## Why section-aware chunking

Most RAG demos split documents by arbitrary token windows. SEC 10-Ks have explicit, legally-defined structure — Item 1A is Risk Factors, Item 7 is MD&A, Item 8 is Financial Statements. Fixed-size chunking destroys that structure, meaning a retrieved chunk about "revenue" gives no signal about whether it came from management's optimistic outlook or the auditor's footnote.

`backend/chunking/chunker.py` chunks by Item section first, then by token budget within each section:

- Items 1, 1A, 7, 7A, and 8 are extracted as first-class sections before any chunking
- Chunks target 500 tokens with 50–75 token overlap
- No chunk crosses a major Item boundary
- Tables are typed and serialized separately, not flattened into prose
- Every chunk carries `ticker`, `year`, `section`, `section_title`, `chunk_type`, and `source_url`
- Chunk IDs are stable and reproducible across runs
- Covered by `backend/tests/test_chunker.py`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Filing source | SEC EDGAR (HTML/XBRL — not PDF) |
| Data validation | Pydantic v2 (all internal contracts) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Local index | JSON + NumPy (`.npy`) — ✅ live for MSFT 2023 |
| Vector store | Pinecone *(Phase 7+)* |
| Keyword search | BM25 *(Phase 7)* |
| Fusion | Reciprocal Rank Fusion *(Phase 7)* |
| Reranking | Planned *(Phase 7)* |
| LLM | OpenAI GPT-4o |
| API | FastAPI — `backend/api/app.py` |
| Frontend | Streamlit — `frontend/apps/streamlit_app.py` |
| Testing | pytest — `backend/tests/` |

> Filings are fetched from SEC EDGAR HTML/XBRL rather than PDFs. HTML preserves section boundaries and avoids the extraction artifacts common in PDF parsing.

---

## Project Structure

```
.
├── backend
│   ├── api/                 # FastAPI app + request/response schemas
│   │   ├── app.py
│   │   └── schemas.py
│   ├── chunking/             # Section-aware chunker
│   │   └── chunker.py
│   ├── ingestion/             # SEC EDGAR client, filing fetch + local cache
│   │   ├── filing_fetcher.py
│   │   └── sec_client.py
│   ├── parsing/               # HTML cleaning, Item section extraction
│   │   └── section_extractor.py
│   ├── retrieval/             # Embeddings, local vector store, retriever
│   │   ├── embedding_client.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── generation/            # Grounded answer generation, citations, uncertainty flagging
│   │   └── answer_generator.py
│   ├── evals/                 # Golden dataset + metrics (Phase 8)
│   ├── data/sec_filings/raw/  # Cached raw filing HTML
│   ├── schemas.py             # Shared Pydantic schemas
│   ├── service.py             # Public interface — the only entry point consumers need
│   └── tests/
│       └── test_chunker.py
├── data
│   ├── index/                 # Local chunk + embedding files
│   └── sec_filings/raw/       # Cached raw filing HTML
├── frontend
│   └── apps/
│       └── streamlit_app.py   # Streamlit demo UI
├── scripts/
│   ├── build_index.py         # Fetch → parse → chunk → embed → save index
│   └── smoke_test_rag.py      # End-to-end pipeline check
├── requirements.txt
└── README.md
```

> Note: raw filing HTML currently lives in both `backend/data/sec_filings/raw/` and `data/sec_filings/raw/`. 

---

## CLI Usage

```bash
# Build the local index for a filing (fetch, parse, chunk, embed, save to data/index/)
python -m scripts.build_index --ticker MSFT --year 2023

# Run an end-to-end smoke test against the local RAG pipeline
python -m scripts.smoke_test_rag
```

*(Exact flags may differ slightly — check `argparse` in each script.)*

Run the API locally:

```bash
uvicorn backend.api.app:app --reload
```

Run the Streamlit demo:

```bash
streamlit run frontend/apps/streamlit_app.py
```

---

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Standalone repo, project structure, Pydantic schemas | ✅ Complete |
| 1 | SEC ingestion — CIK lookup, EDGAR fetch, local filing cache | ✅ Complete |
| 2 | HTML cleaning, section extraction, `FilingSection` objects | ✅ Complete |
| 3 | Section-aware chunking, token budget enforcement, pytest suite | ✅ Complete |
| 4 | Local minimal RAG — embed, store locally, cosine retrieval, grounded answers with citations | ✅ Complete |
| 5 | FastAPI — `/health`, `/ingest`, `/chat`, `/retrieve` | ✅ Complete |
| 6 | Streamlit demo — ticker/year selector, answer display, citation cards, chunk debug view | ✅ Complete |
| 7 | Hybrid retrieval — dense + BM25 + RRF fusion + reranking, Pinecone migration | 🔄 In progress |
| 8 | Eval suite — 20–25 golden prompts, Recall@5, Precision@5, faithfulness scoring | 🔄 In progress |
| 9 | Portfolio polish — README, architecture diagram, screenshots, demo GIF, limitations | 🔄 In progress |

---

## Planned Evaluation

The goal of Phase 8 is to measure whether retrieval improvements actually produce better answers, not just higher similarity scores. The golden dataset will span:

- Factual lookups ("What does MSFT say about cloud revenue?")
- Section-specific questions ("What risks does AAPL disclose in Item 1A?")
- MD&A questions
- Intentionally unanswerable questions (to test hallucination guardrails)
- Year-over-year comparisons

Target output — a results table like:

| Retrieval Strategy | Precision@5 | Citation Accuracy |
|---|---|---|
| Dense only | ~0.58 | — |
| Hybrid + RRF | ~0.72 | — |
| Hybrid + Rerank | ~0.80 | — |

*(Exact numbers will be filled in once Phase 8 is complete.)*

---

## Future: Integration with AI Investment Copilot

Once FinSight is solid as a standalone system, it can be consumed as an external research service by the [AI Investment Copilot](https://github.com/YOUR_USERNAME/ai-investment-copilot).

The copilot's ReAct agent would gain new tools:

```
sec_filing_qa
sec_risk_factor_lookup
sec_mda_lookup
sec_filing_compare
```

This enables blended queries like:

> *"Is it a good time to write a cash-secured put on MSFT?"*

```
Agent:
  1. fetch current MSFT price
  2. fetch options chain
  3. calculate implied / historical volatility
  4. sec_risk_factor_lookup(ticker="MSFT", year="latest")  ← FinSight
  5. synthesize options risk + business risk
```

The agent stops reasoning only from market data — it grounds the trade discussion in audited business risk from the company's own 10-K filing.

---

## Getting Started

> Full Docker setup and deployment notes will be added in Phase 9. The steps below cover local development.

**Prerequisites:** Python 3.11+, OpenAI API key

```bash
git clone https://github.com/YOUR_USERNAME/finsight
cd finsight
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

```bash
# .env
OPENAI_API_KEY=...
SEC_USER_AGENT="Your Name your@email.com"   # required by SEC EDGAR fair-use policy
```

Run the test suite:

```bash
pytest backend/tests/
```

Build the index and run a smoke test:

```bash
python -m scripts.build_index --ticker MSFT --year 2023
python -m scripts.smoke_test_rag
```

---

## Author

**Andrew** · Statistics & Economics, UC Davis  
[LinkedIn](https://linkedin.com/in/andrew-yihanchen) · [GitHub](https://github.com/Andrewchenyh)
