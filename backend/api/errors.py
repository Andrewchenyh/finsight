"""Translate service failures without exposing provider bodies or local paths."""

import httpx
import openai
import requests
from cohere.core.api_error import ApiError as CohereAPIError
from fastapi import HTTPException

from backend.errors import ConfigurationError


def to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, ConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))

    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return HTTPException(
            status_code=503,
            detail="OpenAI access was denied. Check the server's OPENAI_API_KEY and model access.",
        )
    if isinstance(exc, openai.RateLimitError):
        return HTTPException(
            status_code=503,
            detail="OpenAI is rate-limited or out of quota. Check API usage and billing, then retry.",
        )
    if isinstance(exc, openai.APIConnectionError):
        return HTTPException(
            status_code=503,
            detail="OpenAI could not be reached or timed out. Please retry shortly.",
        )
    if isinstance(exc, openai.APIError):
        return HTTPException(
            status_code=502,
            detail="OpenAI could not complete the request. Please retry shortly.",
        )

    cohere_retry = "Retry with retrieval_mode='hybrid' to skip reranking."
    if isinstance(exc, CohereAPIError):
        if exc.status_code in (401, 403, 498):
            return HTTPException(
                status_code=503,
                detail=f"Cohere access was denied. Check the server's COHERE_API_KEY. {cohere_retry}",
            )
        if exc.status_code == 429:
            return HTTPException(
                status_code=503,
                detail=f"Cohere is rate-limited or out of quota. Try again later. {cohere_retry}",
            )
        return HTTPException(
            status_code=502,
            detail=f"Cohere could not complete reranking. {cohere_retry}",
        )
    # Cohere propagates httpx transport failures; OpenAI wraps its own above.
    if isinstance(exc, httpx.RequestError):
        return HTTPException(
            status_code=503,
            detail=f"Cohere could not be reached or timed out. {cohere_retry}",
        )

    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return HTTPException(
            status_code=503,
            detail="SEC EDGAR could not be reached or timed out. Please retry shortly.",
        )
    if isinstance(exc, requests.RequestException):
        status = exc.response.status_code if exc.response is not None else None
        if status in (403, 429):
            return HTTPException(
                status_code=503,
                detail=(
                    "SEC EDGAR denied or limited the request. Check SEC_USER_AGENT "
                    "includes a contact email, then wait before retrying."
                ),
            )
        return HTTPException(
            status_code=502,
            detail="SEC EDGAR returned an unusable response. Please try again later.",
        )

    if isinstance(exc, FileNotFoundError):
        return HTTPException(
            status_code=404,
            detail="The local filing index is unavailable. Build or rebuild it before querying.",
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=400,
            detail=(
                "Invalid request or filing data. Check the query, index name, "
                "ticker, fiscal year, and filters."
            ),
        )
    return HTTPException(
        status_code=500,
        detail="FinSight could not complete the request because of an internal error.",
    )
