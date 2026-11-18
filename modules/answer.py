"""Answer questions from indexed notes with OpenAI and retrieved context."""

import os
import sys
from pathlib import Path

try:
    from modules.config import MODEL, REASONING_EFFORT, create_chat_completion
    from modules.search import search, build_index
except ModuleNotFoundError:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from modules.config import MODEL, REASONING_EFFORT, create_chat_completion
    from modules.search import search, build_index


_SYSTEM_PROMPT = (
    "Answer only from the provided notes. If the notes do not contain the "
    "answer, say so. Cite note numbers like [1]. Keep answers under 120 words."
)
_NOT_FOUND = "I could not find this in your notes."


def answer_question(
    index,
    question,
    top_k=3,
    min_score=0.35,
    model=MODEL,
    reasoning_effort=REASONING_EFFORT,
    api_key=None,
):
    """Answer a question using only sufficiently relevant retrieved notes."""
    sources = search(index, question, top_k=top_k)
    if not sources or sources[0]["score"] < min_score:
        return {"answer": _NOT_FOUND, "sources": []}

    numbered_notes = "\n\n".join(
        f"[{number}] {source['text']}"
        for number, source in enumerate(sources, start=1)
    )
    user_prompt = f"Notes:\n{numbered_notes}\n\nQuestion: {question}"

    try:
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)
        response = create_chat_completion(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = response.choices[0].message.content or ""
        answer_text = answer_text.strip()
        normalized_answer = (
            answer_text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .casefold()
        )
        refusal_phrases = (
            "do not say",
            "does not say",
            "don't provide",
            "do not provide",
            "not provide",
            "could not find",
            "not contain",
            "don't contain",
            "do not contain",
            "doesn't contain",
            "does not contain",
            "not in the notes",
            "not mentioned in the notes",
            "not stated in the notes",
        )
        if any(phrase in normalized_answer for phrase in refusal_phrases):
            return {"answer": answer_text, "sources": []}
        return {"answer": answer_text, "sources": sources}
    except Exception:
        return {
            "answer": "Sorry, I could not generate an answer. Check your API setup and try again.",
            "sources": sources,
        }


if __name__ == "__main__":
    sample_chunks = [
        "Planets orbit stars, and telescopes help astronomers study distant galaxies.",
        "Roasting vegetables with olive oil and herbs makes a simple flavorful meal.",
        "Football players pass the ball, defend their goal, and work as a team.",
        "Python functions organize reusable code and can return calculated values.",
        "Ocean currents move warm and cold water through habitats filled with marine life.",
    ]
    sample_index = build_index(sample_chunks)
    questions = [
        "How do astronomers study distant galaxies?",
        "Who won the 1998 cricket world cup?",
    ]

    for sample_question in questions:
        print(answer_question(sample_index, sample_question))
