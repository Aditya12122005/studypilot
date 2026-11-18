from modules import answer


def test_answer_refusal_is_returned_without_sources(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = fake_openai(["I could not find this in your notes."])
    monkeypatch.setattr(
        answer,
        "search",
        lambda index, question, top_k: [{"text": "Some note", "score": 0.9}],
    )

    result = answer.answer_question({}, "What is missing?")

    assert result == {
        "answer": "I could not find this in your notes.",
        "sources": [],
    }
    assert len(calls) == 1
