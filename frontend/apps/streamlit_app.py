import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


DEFAULT_API_URL = "http://127.0.0.1:8000"
SECTION_OPTIONS = [
    "Item 1",
    "Item 1A",
    "Item 7",
    "Item 7A",
    "Item 8",
]


def get_api_url() -> str:
    return os.getenv("FINSIGHT_API_URL", DEFAULT_API_URL).rstrip("/")


def call_chat_api(
    api_url: str,
    query: str,
    index_name: str,
    ticker: str,
    fiscal_year: int,
    section: str,
    top_k: int,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "index_name": index_name,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "section": section,
        "top_k": top_k,
    }

    response = requests.post(
        f"{api_url}/chat",
        json=payload,
        timeout=90,
    )
    response.raise_for_status()

    return response.json()


def call_health_api(api_url: str) -> bool:
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False
    

def call_ingest_api(
    api_url: str,
    ticker: str,
    fiscal_year: int,
    index_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
    }

    if index_name:
        payload["index_name"] = index_name

    response = requests.post(
        f"{api_url}/ingest",
        json=payload,
        timeout=240,
    )
    response.raise_for_status()

    return response.json()


def render_citations(citations: list[dict[str, Any]]) -> None:
    st.subheader("Citations")

    if not citations:
        st.info("No citations returned.")
        return

    for citation in citations:
        citation_id = citation["citation_id"]
        ticker = citation["ticker"]
        fiscal_year = citation["fiscal_year"]
        section = citation["section"]
        section_title = citation["section_title"]
        source_url = citation["source_url"]
        excerpt = citation["excerpt"]

        with st.expander(f"[{citation_id}] {ticker} {fiscal_year} {section} - {section_title}"):
            st.markdown(f"[Open SEC filing]({source_url})")
            st.write(excerpt)


def render_retrieved_chunks(retrieved_chunks: list[dict[str, Any]]) -> None:
    st.subheader("Retrieved Chunks")

    if not retrieved_chunks:
        st.info("No retrieved chunks returned.")
        return

    for item in retrieved_chunks:
        chunk = item["chunk"]
        rank = item["rank"]
        score = item["score"]
        method = item["retrieval_method"]

        label = (
            f"Rank {rank} | Score {score:.4f} | "
            f"{chunk['section']} | {chunk['chunk_id']}"
        )

        with st.expander(label):
            st.caption(f"Retrieval method: {method}")
            st.write(chunk["text"])


def main() -> None:
    st.set_page_config(
        page_title="FinSight",
        page_icon="",
        layout="wide",
    )

    api_url = get_api_url()

    st.title("FinSight")
    st.caption("SEC 10-K Q&A with grounded citations")

    with st.sidebar:
        st.header("Query Settings")

        api_online = call_health_api(api_url)
        if api_online:
            st.success("API connected")
        else:
            st.error("API unavailable")

        ticker = st.text_input("Ticker", value="MSFT").upper().strip()
        fiscal_year = st.number_input(
            "Fiscal year",
            min_value=1994,
            max_value=2100,
            value=2023,
            step=1,
        )
        section = st.selectbox("10-K section", SECTION_OPTIONS, index=1)
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)

        index_name = st.text_input(
            "Index name",
            value=f"{ticker}_{fiscal_year}",
            help="Local index name built by the API or scripts.",
        )
        rebuild_index = st.button(
            "Build / Rebuild Index",
            disabled=not api_online or not ticker,
            help="Fetch the filing, extract sections, embed chunks, and save the local index.",
        )

        if rebuild_index:
            with st.spinner(f"Building index {index_name}..."):
                try:
                    ingest_result = call_ingest_api(
                        api_url=api_url,
                        ticker=ticker,
                        fiscal_year=int(fiscal_year),
                        index_name=index_name,
                    )
                    st.success(ingest_result["message"])
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Ingest failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Ingest request failed: {exc}")
                    
        st.divider()
        st.caption(f"API: {api_url}")

    query = st.text_area(
        "Ask a question about the filing",
        value="What cybersecurity risks does Microsoft describe?",
        height=100,
    )

    submitted = st.button("Ask FinSight", type="primary", disabled=not query.strip())

    if submitted:
        if not api_online:
            st.error("FastAPI backend is not running. Start it with `uvicorn backend.api.app:app --reload`.")
            return

        with st.spinner("Retrieving filing evidence and generating answer..."):
            try:
                payload = call_chat_api(
                    api_url=api_url,
                    query=query,
                    index_name=index_name,
                    ticker=ticker,
                    fiscal_year=int(fiscal_year),
                    section=section,
                    top_k=top_k,
                )
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"API error: {detail}")
                return
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        result = payload["result"]

        st.subheader("Answer")
        st.markdown(result["answer"])

        left, right = st.columns([1, 1])

        with left:
            render_citations(result["citations"])

        with right:
            render_retrieved_chunks(result["retrieved_chunks"])

        if result.get("limitations"):
            st.subheader("Limitations")
            for limitation in result["limitations"]:
                st.warning(limitation)


if __name__ == "__main__":
    main()