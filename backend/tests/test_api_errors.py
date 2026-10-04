from unittest.mock import Mock

import httpx
import openai
import pytest
import requests
from cohere.core.api_error import ApiError as CohereApiError
from fastapi.testclient import TestClient

from backend.api import app as api
from backend.errors import ConfigurationError
from backend.generation.answer_generator import AnswerGenerator
from backend.ingestion.filing_fetcher import FilingFetcher
from backend.ingestion.sec_client import SECClient
from backend.retrieval.embedding_client import EmbeddingClient
from backend.retrieval.reranker import CohereReranker


SECRET = "fake-secret-sentinel-do-not-expose"
PRIVATE_PATH = "/private/local-machine/secret-index.json"
PRIVATE_DETAIL = f"{SECRET} {PRIVATE_PATH}"
ENDPOINTS = {
    "/retrieve": ("retrieve_sec_chunks", {"query": "Risks?", "index_name": "MSFT_2023"}),
    "/chat": ("answer_sec_question", {"query": "Risks?", "index_name": "MSFT_2023"}),
    "/ingest": ("build_sec_index", {"ticker": "MSFT", "fiscal_year": 2023}),
}
PROVIDER_REQUEST = httpx.Request(
    "POST", f"https://provider.example.test/?key={SECRET}"
)


def openai_status_error(error_type, status: int):
    response = httpx.Response(status, request=PROVIDER_REQUEST)
    return error_type(PRIVATE_DETAIL, response=response, body={"message": PRIVATE_DETAIL})


def sec_http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.url = f"https://www.sec.gov/?key={SECRET}"
    return requests.HTTPError(PRIVATE_DETAIL, response=response)


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def request_with_failure(client, monkeypatch, endpoint: str, error: Exception):
    function_name, payload = ENDPOINTS[endpoint]
    service = Mock(side_effect=error)
    monkeypatch.setattr(api, function_name, service)
    response = client.post(endpoint, json=payload)
    service.assert_called_once()
    assert SECRET not in response.text
    assert PRIVATE_PATH not in response.text
    assert isinstance(response.json()["detail"], str)
    return response


@pytest.mark.parametrize(
    "endpoint,error,status,provider",
    [
        pytest.param("/retrieve", openai_status_error(openai.AuthenticationError, 401), 503, "OpenAI", id="openai-auth"),
        pytest.param("/chat", openai_status_error(openai.RateLimitError, 429), 503, "OpenAI", id="openai-rate-limit"),
        pytest.param("/ingest", openai.APITimeoutError(PROVIDER_REQUEST), 503, "OpenAI", id="openai-timeout"),
        pytest.param("/retrieve", openai.APIConnectionError(message=PRIVATE_DETAIL, request=PROVIDER_REQUEST), 503, "OpenAI", id="openai-connection"),
        pytest.param("/chat", openai.APIError(PRIVATE_DETAIL, PROVIDER_REQUEST, body=PRIVATE_DETAIL), 502, "OpenAI", id="openai-response"),
        pytest.param("/chat", CohereApiError(status_code=498, body=PRIVATE_DETAIL), 503, "Cohere", id="cohere-invalid-token"),
        pytest.param("/retrieve", CohereApiError(status_code=429, body=PRIVATE_DETAIL), 503, "Cohere", id="cohere-rate-limit"),
        pytest.param("/chat", CohereApiError(status_code=500, body=PRIVATE_DETAIL), 502, "Cohere", id="cohere-response"),
        pytest.param("/chat", httpx.ReadTimeout(PRIVATE_DETAIL, request=PROVIDER_REQUEST), 503, "Cohere", id="cohere-timeout"),
        pytest.param("/retrieve", httpx.ConnectError(PRIVATE_DETAIL, request=PROVIDER_REQUEST), 503, "Cohere", id="cohere-connection"),
        pytest.param("/ingest", sec_http_error(403), 503, "SEC", id="sec-forbidden"),
        pytest.param("/ingest", sec_http_error(429), 503, "SEC", id="sec-rate-limit"),
        pytest.param("/ingest", requests.Timeout(PRIVATE_DETAIL), 503, "SEC", id="sec-timeout"),
    ],
)
def test_provider_errors_are_safe_and_actionable(
    client, monkeypatch, endpoint, error, status, provider
) -> None:
    response = request_with_failure(client, monkeypatch, endpoint, error)

    assert response.status_code == status
    detail = response.json()["detail"]
    assert provider.lower() in detail.lower()
    if provider == "Cohere":
        assert "hybrid" in detail


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "error_type,status",
    [(RuntimeError, 500), (ValueError, 400), (FileNotFoundError, 404)],
)
def test_internal_errors_do_not_expose_exception_details(
    client, monkeypatch, endpoint, error_type, status
) -> None:
    response = request_with_failure(
        client, monkeypatch, endpoint, error_type(PRIVATE_DETAIL)
    )

    assert response.status_code == status


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_configuration_errors_preserve_only_the_public_message(
    client, monkeypatch, endpoint
) -> None:
    public_message = "OPENAI_API_KEY is not set. Configure it before retrying."
    response = request_with_failure(
        client, monkeypatch, endpoint, ConfigurationError(public_message)
    )

    assert response.status_code == 503
    assert response.json()["detail"] == public_message


@pytest.mark.parametrize(
    "constructor,variable,client_path",
    [
        (EmbeddingClient, "OPENAI_API_KEY", "backend.retrieval.embedding_client.OpenAI"),
        (AnswerGenerator, "OPENAI_API_KEY", "backend.generation.answer_generator.OpenAI"),
        (CohereReranker, "COHERE_API_KEY", "backend.retrieval.reranker.cohere.ClientV2"),
        (SECClient, "SEC_USER_AGENT", "backend.ingestion.sec_client.requests.Session"),
        (FilingFetcher, "SEC_USER_AGENT", "backend.ingestion.filing_fetcher.requests.Session"),
    ],
)
def test_real_constructors_reject_missing_configuration_before_client_creation(
    monkeypatch, tmp_path, constructor, variable, client_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(variable, raising=False)
    external_client = Mock(side_effect=AssertionError("Client must not be initialized"))
    monkeypatch.setattr(client_path, external_client)

    with pytest.raises(ConfigurationError, match=variable) as error:
        constructor()

    external_client.assert_not_called()
    if constructor is CohereReranker:
        assert "hybrid" in str(error.value)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_request_validation_still_returns_422(client, monkeypatch, endpoint) -> None:
    function_name, _ = ENDPOINTS[endpoint]
    service = Mock()
    monkeypatch.setattr(api, function_name, service)

    response = client.post(endpoint, json={})

    assert response.status_code == 422
    service.assert_not_called()


def test_health_remains_available(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "finsight"}
