import numpy as np

from modules import search


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_search_returns_closest_chunk_on_tiny_index(monkeypatch):
    monkeypatch.setattr(search, "_get_model", lambda: FakeEmbeddingModel())
    index = {
        "chunks": ["the target answer", "an unrelated passage"],
        "embeddings": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }

    results = search.search(index, "query", top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "the target answer"
    assert results[0]["chunk_id"] == 0
