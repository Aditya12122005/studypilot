"""End-to-end check: does the full pipeline refuse off-topic questions?"""
import os

from modules.loader import clean_text, split_into_chunks
from modules.search import build_index
from modules.answer import answer_question

# Replace these with the off-topic questions from your evaluate.py if they differ.
OFF_TOPIC = [
    "Who won the 1998 cricket world cup?",
    "How do I bake sourdough bread?",
    "What is the capital of Australia?",
    "How many moons does the Sun have?",
    "What is the temperature on Jupiter's largest moon?",
    "How long does Saturn take to orbit the Sun in Earth days?",
    "How far is the Moon from Mars?",
    "Who invented the telephone?",
]

MARKERS = (
    "could not find", "cannot find", "can't find", "not contain",
    "do not contain", "does not contain", "don't contain", "not mention",
    "no information", "not in the notes", "not provided", "not say", "n\x27t say", "not provide", "n\x27t provide",
)


def refused(answer):
    text = answer.lower().replace("\u2019", "\x27")
    return any(marker in text for marker in MARKERS)


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in this PowerShell window first.")
        return
    with open("sample_notes.txt", encoding="utf-8") as handle:
        notes = clean_text(handle.read())
    index = build_index(split_into_chunks(notes, chunk_size=300, overlap=100))

    refusals = 0
    for question in OFF_TOPIC:
        result = answer_question(index, question)
        ok = refused(result["answer"])
        refusals += ok
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"Refused: {ok}\n")
    print(f"Refusal rate: {refusals}/{len(OFF_TOPIC)}")


if __name__ == "__main__":
    main()

