import json

from modules.studykit import make_flashcards, make_quiz


def test_flashcards_accept_valid_json_in_code_fence(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = fake_openai(
        ['```json\n' + json.dumps([{"front": "  Term  ", "back": "  Meaning  "}]) + '\n```']
    )

    cards = make_flashcards("notes", n=1)

    assert cards == [{"front": "Term", "back": "Meaning"}]
    assert len(calls) == 1


def test_quiz_rejects_boolean_answer_index(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    invalid_quiz = [{
        "question": "Question?",
        "options": ["A", "B", "C", "D"],
        "answer_index": True,
        "explanation": "Because.",
    }]
    calls = fake_openai([json.dumps(invalid_quiz), json.dumps(invalid_quiz)])

    assert make_quiz("notes", n=1) == []
    assert len(calls) == 2
