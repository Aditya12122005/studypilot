"""Keyword extraction and extractive summarization helpers."""

from pathlib import Path

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


_NLTK_DATA_DIR = Path(__file__).resolve().parent / "nltk_data"
if str(_NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))


def _ensure_nltk_data():
    """Download the sentence tokenizer resources into this project if needed."""
    resources = (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab/english/"),
    )
    for package, resource in resources:
        try:
            nltk.data.find(resource, paths=[str(_NLTK_DATA_DIR)])
        except LookupError:
            if not nltk.download(package, download_dir=str(_NLTK_DATA_DIR), quiet=True):
                raise LookupError(f"Could not download NLTK resource: {package}")


def _sentences(text):
    _ensure_nltk_data()
    return nltk.sent_tokenize(str(text))


def extract_keywords(text, top_n=8):
    """Return the top terms ranked by their summed sentence TF-IDF scores."""
    sentences = _sentences(text)
    if not sentences or top_n <= 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        scores = vectorizer.fit_transform(sentences).sum(axis=0).A1
    except ValueError:
        return []

    terms = vectorizer.get_feature_names_out()
    ranked_terms = sorted(
        zip(terms, scores), key=lambda item: (-float(item[1]), item[0])
    )
    return [term for term, _ in ranked_terms[:top_n]]


def summarize(text, num_sentences=3):
    """Return the highest-scoring sentences in their original order."""
    sentences = _sentences(text)
    if not sentences or num_sentences <= 0:
        return ""

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        return ""

    vocabulary = vectorizer.vocabulary_
    analyze = vectorizer.build_analyzer()
    sentence_scores = []

    for index, sentence in enumerate(sentences):
        row = sentence_vectors.getrow(index)
        weights = dict(zip(row.indices, row.data))
        word_indices = [
            vocabulary[word] for word in analyze(sentence) if word in vocabulary
        ]
        average_score = (
            sum(weights.get(word_index, 0.0) for word_index in word_indices)
            / len(word_indices)
            if word_indices
            else 0.0
        )
        sentence_scores.append((index, average_score))

    top_indices = {
        index
        for index, _ in sorted(
            sentence_scores, key=lambda item: (-item[1], item[0])
        )[:num_sentences]
    }
    return " ".join(
        sentence for index, sentence in enumerate(sentences) if index in top_indices
    )


if __name__ == "__main__":
    sample = (
        "StudyPilot helps students organize course notes. "
        "It extracts useful keywords from each document. "
        "TF-IDF scores help identify important terms. "
        "Sentence ranking can create a short extractive summary. "
        "PDF text can be cleaned before analysis. "
        "These tools make review sessions more focused."
    )
    print("Keywords:", extract_keywords(sample))
    print("Summary:", summarize(sample))
