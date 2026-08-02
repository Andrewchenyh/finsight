# FinSight

![CI](https://github.com/Andrewchenyh/finsight/actions/workflows/ci.yml/badge.svg)

> A RAG-powered SEC 10-K research assistant. Ask grounded questions over real company filings and verify answers with citations traced back to filing sections.

FinSight is a standalone AI research tool for SEC filings. It ingests 10-K filings from SEC EDGAR, extracts major filing sections, builds a local retrieval index, retrieves relevant filing passages, and generates citation-backed answers.

It is also designed so it can later plug into a broader [AI Investment Copilot](https://github.com/Andrewchenyh/ai-investment-copilot) as a dedicated filing research service.

---

## What It Does

Ask a question like:

> What cybersecurity risks does Microsoft describe in its 2023 10-K?

FinSight:

1. Retrieves relevant passages from the indexed 10-K
2. Generates a grounded answer using only retrieved filing excerpts
3. Returns citations with ticker, year, filing type, section, source URL, and excerpt
4. Exposes the workflow through scripts, FastAPI, and a Streamlit demo UI

Example answer shape:

```text
Answer:
Microsoft describes cybersecurity risks including evolving threats, vulnerabilities
in products and services, possible data breaches, supply chain cyberattacks, and
customer reliance on cloud infrastructure security [1][2][3].

Citations:
[1] MSFT 2023 10-K - Item 1A - Risk Factors
[2] MSFT 2023 10-K - Item 1A - Risk Factors
[3] MSFT 2023 10-K - Item 1A - Risk Factors
```

---

## Architecture

```text
Scripts / FastAPI / Streamlit UI
              |
              v
      FinSight Service Layer
              |
      +-------+--------+
      |                |
      v                v
 Ingestion         Retrieval + Generation
 Pipeline          Pipeline
      |                |
      v                v
 SEC EDGAR       Local Retrieval Index
 10-K HTML       JSON chunks + NumPy embeddings
```

Detailed flow:

```text
ticker + fiscal year
  -> CIK lookup
  -> SEC submissions metadata
  -> 10-K source URL
  -> raw HTML download/cache
  -> clean filing text
  -> extract Item sections
  -> section-aware chunks
  -> OpenAI embeddings
  -> local vector index
  -> dense/BM25/hybrid retrieval
  -> optional Cohere rerank
  -> grounded answer generation
  -> citations
```

---

## Why Section-Aware Chunking

Most RAG demos split documents by arbitrary token windows. SEC 10-Ks have legally meaningful structure:

- Item 1: Business
- Item 1A: Risk Factors
- Item 7: Management's Discussion and Analysis
- Item 7A: Market Risk
- Item 8: Financial Statements

FinSight extracts these sections first, then chunks within each section. This preserves provenance and makes citations more useful.

Each chunk carries metadata such as:

```text
ticker
company
CIK
accession number
fiscal year
filing type
section
section title
source URL
character offsets
estimated token count
```

Chunking behavior is covered by `backend/tests/test_chunker.py`.

---

## Retrieval Modes

FinSight supports four retrieval modes:

| Mode | Description |
|---|---|
| `dense` | OpenAI embedding retrieval over local chunk vectors |
| `bm25` | Keyword-based lexical retrieval |
| `hybrid` | Dense + BM25 candidate fusion with Reciprocal Rank Fusion |
| `hybrid_rerank` | Hybrid candidate retrieval followed by Cohere reranking |

`hybrid_rerank` is the quality-oriented default. `dense` remains useful as a faster, cheaper baseline.

---

## Retrieval Evaluation

FinSight includes a starter 10-question retrieval benchmark over the Microsoft 2023 10-K. The benchmark covers risk factors, MD&A, business competition, revenue recognition, market risk, and datacenter dependencies.

Metrics:

- `Recall@5`: whether at least one expected chunk appeared in the top 5
- `Mean Precision@5`: approximate relevance rate across retrieved chunks
- `Mean Expected Chunk Rank`: average rank of the first expected chunk hit

| Retrieval Mode | Recall@5 | Mean Precision@5 | Mean Expected Chunk Rank |
|---|---:|---:|---:|
| Dense | 0.90 | 0.58 | 1.56 |
| BM25 | 1.00 | 0.50 | 2.10 |
| Hybrid | 1.00 | 0.60 | 1.70 |
| Hybrid + Rerank | 1.00 | 0.70 | 1.40 |

Takeaway: hybrid retrieval with Cohere reranking produced the strongest overall retrieval quality, improving Mean Precision@5 from `0.58` to `0.70` versus dense retrieval while maintaining perfect Recall@5 on the starter benchmark. Reranking was especially useful for MD&A queries such as Economic Conditions and Foreign Exchange impacts, moving both target chunks to rank 1.

Run the eval:

```bash
python -m scripts.run_retrieval_eval
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Filing source | SEC EDGAR HTML/XBRL |
| Backend | Python |
| Data validation | Pydantic v2 |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini` |
| Dense retrieval | NumPy dot-product search over local vectors |
| Lexical retrieval | BM25 via `rank-bm25` |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Cohere Rerank |
| Local index | JSON + NumPy `.npy` |
| API | FastAPI |
| Frontend | Streamlit |
| Tests/CI | pytest + GitHub Actions |

Filings are fetched from SEC EDGAR HTML/XBRL rather than PDFs. HTML preserves section boundaries and avoids many PDF extraction artifacts.

---

## Project Structure

```text
backend/
  api/
    app.py                 # FastAPI app
    schemas.py             # API request/response models

  chunking/
    chunker.py             # Section-aware chunking

  evals/
    schemas.py             # Eval dataset/result models
    retrieval_metrics.py   # Retrieval metric helpers

  generation/
    answer_generator.py    # Grounded answer generation

  ingestion/
    sec_client.py          # SEC ticker/CIK/submissions client
    filing_fetcher.py      # Raw filing download/cache

  parsing/
    section_extractor.py   # HTML cleaning + 10-K section extraction

  retrieval/
    embedding_client.py    # OpenAI embedding wrapper
    retriever.py           # DenseRetriever
    bm25_retriever.py      # BM25Retriever
    hybrid_retriever.py    # Dense + BM25 RRF fusion
    reranker.py            # Cohere reranker
    rerank_retriever.py    # Hybrid + rerank wrapper
    vector_store.py        # Local JSON + NumPy vector store

  schemas.py               # Shared Pydantic models
  service.py               # Public service functions
  tests/                   # Unit tests

frontend/
  apps/
    streamlit_app.py       # Streamlit demo UI

scripts/
  build_index.py           # Build local index for ticker/year
  smoke_test_rag.py        # End-to-end RAG smoke test
  retrieval_baseline.py    # Dense baseline inspection
  compare_retrieval_modes.py
  run_retrieval_eval.py    # Starter benchmark runner

data/
  evals/                   # Golden retrieval datasets, committed
  sec_filings/raw/         # Cached raw filings, not committed
  index/                   # Local chunks + embeddings, not committed
```

---

## Setup

Prerequisites:

- Python 3.11+
- OpenAI API key
- Cohere API key for reranking
- SEC-compliant user agent string

```bash
git clone https://github.com/Andrewchenyh/finsight
cd finsight

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
SEC_USER_AGENT="FinSight your_email@example.com"
FINSIGHT_API_URL=http://127.0.0.1:8000
```

---

## Build A Local Index

Build the Microsoft 2023 10-K index:

```bash
python -m scripts.build_index --ticker MSFT --year 2023
```

This creates:

```text
data/index/MSFT_2023_chunks.json
data/index/MSFT_2023_embeddings.npy
```

---

## Run The API

```bash
uvicorn backend.api.app:app --reload
```

Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```text
GET  /health
POST /ingest
POST /retrieve
POST /chat
```

Example chat request:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What cybersecurity risks does Microsoft describe?",
    "index_name": "MSFT_2023",
    "ticker": "MSFT",
    "fiscal_year": 2023,
    "section": "Item 1A",
    "top_k": 5,
    "retrieval_mode": "hybrid_rerank"
  }'
```

---

## Run The Streamlit Demo

Start the FastAPI backend first:

```bash
uvicorn backend.api.app:app --reload
```

Then in another terminal:

```bash
streamlit run frontend/apps/streamlit_app.py
```

The UI includes:

- ticker selector
- fiscal year selector
- 10-K section filter
- retrieval mode selector
- index build/rebuild button
- question input
- generated answer
- citation cards
- retrieval debug view

---

## Tests And CI

Run deterministic tests:

```bash
python -m pytest
```

Run compile checks locally if needed:

```bash
python -m py_compile backend/schemas.py backend/service.py backend/api/schemas.py backend/api/app.py
```

Run optional live smoke tests:

```bash
python -m scripts.smoke_test_rag
python -m scripts.compare_retrieval_modes
python -m scripts.run_retrieval_eval
```

Live smoke/eval scripts call OpenAI and/or Cohere and require local indexes plus API keys.

---

## Future Integration With AI Investment Copilot

Once FinSight is solid as a standalone product, it can be consumed by the AI Investment Copilot as an external SEC filing research service.

Potential tools:

```text
sec_filing_qa
sec_risk_factor_lookup
sec_mda_lookup
sec_filing_compare
```

Example blended query:

> Is it a good time to write a cash-secured put on MSFT?

Possible agent flow:

```text
1. Fetch current MSFT price
2. Fetch options chain
3. Calculate volatility and risk/reward
4. Ask FinSight for latest MSFT 10-K risk factors
5. Synthesize market risk + business risk
```

This lets the copilot ground investment analysis in both market data and audited company disclosures.

---

## Limitations

- Starter evaluation currently covers MSFT 2023 only
- `/ingest` is synchronous
- No production auth yet
- No Docker/deployment setup yet
- Citation excerpts are chunk-based, not exact sentence-level spans
- Hybrid rerank uses external APIs and is slower/costlier than dense mode

---

## Author

Andrew Chen  
Statistics & Economics, UC Davis  

[LinkedIn](https://linkedin.com/in/andrew-yihanchen) · [GitHub](https://github.com/Andrewchenyh)
