"""In-memory semantic search using sentence-transformer embeddings."""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_CACHE_DIR = Path(__file__).resolve().parent / "model_cache"
_MODEL = None


def _get_model():
    """Load the sentence-transformer model once and reuse it."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            _MODEL_NAME, cache_folder=str(_MODEL_CACHE_DIR)
        )
    return _MODEL


def build_index(chunks, embeddings=None):
    """Build embedding and keyword indexes, reusing embeddings when supplied."""
    chunk_records = [
        chunk if isinstance(chunk, dict) else {"text": chunk}
        for chunk in chunks
    ]
    chunk_texts = [record["text"] for record in chunk_records]
    if chunk_texts:
        if embeddings is None:
            embeddings = _get_model().encode(
                chunk_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk_texts):
                raise ValueError("embeddings must have one row per chunk")
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    vectorizer = TfidfVectorizer()
    try:
        keyword_vectors = vectorizer.fit_transform(chunk_texts)
    except ValueError:
        vectorizer = None
        keyword_vectors = None

    return {
        "chunks": chunk_texts,
        "chunk_pages": [
            (record.get("page_start"), record.get("page_end"))
            for record in chunk_records
        ],
        "embeddings": embeddings,
        "keyword_vectorizer": vectorizer,
        "keyword_vectors": keyword_vectors,
    }


def _embedding_scores(index, query):
    chunks = index.get("chunks", [])
    embeddings = np.asarray(index.get("embeddings", []), dtype=np.float32)
    query_embedding = _get_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(embeddings, axis=1)
    denominators = chunk_norms * query_norm
    scores = np.divide(
        embeddings @ query_embedding,
        denominators,
        out=np.zeros(len(chunks), dtype=np.float32),
        where=denominators != 0,
    )

    return scores


def _keyword_scores(index, query):
    chunks = index.get("chunks", [])
    vectorizer = index.get("keyword_vectorizer")
    keyword_vectors = index.get("keyword_vectors")
    if not chunks or vectorizer is None or keyword_vectors is None:
        return np.zeros(len(chunks), dtype=np.float32)
    query_vector = vectorizer.transform([query])
    if query_vector.nnz == 0:
        return np.zeros(len(chunks), dtype=np.float32)
    return np.asarray((keyword_vectors @ query_vector.T).toarray()).ravel().astype(
        np.float32
    )


def _format_results(index, scores, top_k, include_component_scores=False,
                    embedding_scores=None, keyword_scores=None):
    chunks = index.get("chunks", [])
    if top_k <= 0 or not chunks:
        return []
    ranked_ids = np.argsort(-scores, kind="stable")[: min(top_k, len(chunks))]
    chunk_pages = index.get("chunk_pages", [(None, None)] * len(chunks))
    results = []
    for chunk_id in ranked_ids:
        chunk_id = int(chunk_id)
        result = {
            "text": chunks[int(chunk_id)],
            "score": float(scores[chunk_id]),
            "chunk_id": int(chunk_id),
            "page_start": chunk_pages[int(chunk_id)][0],
            "page_end": chunk_pages[int(chunk_id)][1],
        }
        if include_component_scores:
            result["embedding_score"] = float(embedding_scores[chunk_id])
            result["keyword_score"] = float(keyword_scores[chunk_id])
        results.append(result)
    return results


def search(index, query, top_k=3):
    """Return the most similar chunks ranked by embedding cosine similarity."""
    chunks = index.get("chunks", [])
    embeddings = np.asarray(index.get("embeddings", []), dtype=np.float32)
    if top_k <= 0 or not chunks or embeddings.size == 0:
        return []
    scores = _embedding_scores(index, query)
    return _format_results(index, scores, top_k)


def hybrid_search(index, query, top_k=3, alpha=0.7):
    """Rank chunks by alpha * embedding similarity + (1-alpha) * TF-IDF."""
    if not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    chunks = index.get("chunks", [])
    embeddings = np.asarray(index.get("embeddings", []), dtype=np.float32)
    if top_k <= 0 or not chunks or embeddings.size == 0:
        return []
    embedding_scores = _embedding_scores(index, query)
    keyword_scores = _keyword_scores(index, query)
    scores = alpha * embedding_scores + (1.0 - alpha) * keyword_scores
    return _format_results(
        index,
        scores,
        top_k,
        include_component_scores=True,
        embedding_scores=embedding_scores,
        keyword_scores=keyword_scores,
    )


if __name__ == "__main__":
    sample_chunks = [
        "Planets orbit stars, and telescopes help astronomers study distant galaxies.",
        "Roasting vegetables with olive oil and herbs makes a simple flavorful meal.",
        "Football players pass the ball, defend their goal, and work as a team.",
        "Python functions organize reusable code and can return calculated values.",
        "Ocean currents move warm and cold water through habitats filled with marine life.",
    ]
    index = build_index(sample_chunks)
    queries = [
        "How do astronomers study stars and planets?",
        "How can I organize reusable code in Python?",
    ]

    for query in queries:
        results = search(index, query, top_k=3)
        if results:
            print(f"Query: {query}")
            print(f"Top result: {results[0]['text']}")
            print(f"Score: {results[0]['score']:.4f}")
