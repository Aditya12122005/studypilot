from modules import analysis


SENTENCES = [
    "Fiber helps digestion.",
    "Apples provide vitamin C.",
    "Beans are rich in fiber and protein.",
]
SAMPLE = " | ".join(SENTENCES)


def test_extract_keywords_from_fixed_sample(monkeypatch):
    monkeypatch.setattr(analysis, "_sentences", lambda text: SENTENCES)

    keywords = analysis.extract_keywords(SAMPLE, top_n=3)

    assert len(keywords) == 3
    assert "fiber" in keywords


def test_summarize_selects_fixed_sample_sentences_in_source_order(monkeypatch):
    monkeypatch.setattr(analysis, "_sentences", lambda text: SENTENCES)

    summary = analysis.summarize(SAMPLE, num_sentences=2)
    selected = [sentence for sentence in SENTENCES if sentence in summary]

    assert summary
    assert len(selected) == 2
    assert summary == " ".join(selected)
