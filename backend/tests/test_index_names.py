from pathlib import Path
from unittest.mock import Mock

import pytest

from backend import service
from backend.retrieval.vector_store import LocalVectorStore


@pytest.mark.parametrize("index_name", ["MSFT_2023", "test-MSFT_2023", "BRK.B_2023"])
def test_valid_index_names_stay_inside_index_directory(
    tmp_path: Path, index_name: str
) -> None:
    index_dir = tmp_path / "index"

    store = LocalVectorStore(index_name=index_name, index_dir=index_dir)

    assert store.chunks_path.resolve() == index_dir.resolve() / f"{index_name}_chunks.json"
    assert store.embeddings_path.resolve() == index_dir.resolve() / f"{index_name}_embeddings.npy"


@pytest.mark.parametrize(
    "index_name",
    ["../outside", "nested/index", r"..\outside", "/tmp/outside", r"C:\outside", "", ".."],
)
def test_invalid_index_names_are_rejected_before_directory_creation(
    tmp_path: Path, index_name: str
) -> None:
    index_dir = tmp_path / "index"

    with pytest.raises(ValueError, match="index_name"):
        LocalVectorStore(index_name=index_name, index_dir=index_dir)

    assert not index_dir.exists()


def test_ingest_rejects_invalid_index_name_before_external_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    clients = {}
    for name in ("SECClient", "FilingFetcher", "EmbeddingClient"):
        client = Mock(side_effect=AssertionError(f"{name} must not be initialized"))
        monkeypatch.setattr(service, name, client)
        clients[name] = client

    with pytest.raises(ValueError, match="index_name"):
        service.build_sec_index(ticker="MSFT", fiscal_year=2023, index_name="../outside")

    for client in clients.values():
        client.assert_not_called()
    assert not (tmp_path / "data").exists()
