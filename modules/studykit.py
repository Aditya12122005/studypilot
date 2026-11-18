"""Generate validated study aids from supplied text."""

import json
import os
import sys
from pathlib import Path

try:
    from modules.config import MODEL, REASONING_EFFORT, create_chat_completion
except ModuleNotFoundError:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from modules.config import MODEL, REASONING_EFFORT, create_chat_completion

_SYSTEM_PROMPT = "Use only the given text; return only valid JSON with no markdown."


def _strip_code_fences(reply):
    reply = reply.strip()
    if reply.startswith("```"):
        lines = reply.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        reply = "\n".join(lines).strip()
    return reply


def _valid_count(n):
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def _generate_list(text, n, model, kind, reasoning_effort, api_key):
    if not _valid_count(n) or n == 0:
        return []

    if kind == "flashcards":
        shape = '[{"front": "...", "back": "..."}]'
        requirements = (
            f"Create exactly {n} flashcards. Each item must have non-empty string "
            "fields named front and back."
        )
    else:
        shape = (
            '[{"question": "...", "options": ["...", "...", "...", "..."], '
            '"answer_index": 0, "explanation": "..."}]'
        )
        requirements = (
            f"Create exactly {n} multiple-choice questions. Each item must have a "
            "non-empty string question, exactly four string options, an integer "
            "answer_index from 0 to 3, and a non-empty string explanation."
        )

    user_prompt = (
        f"Given text:\n{text}\n\n{requirements} "
        f"Return a JSON array in this shape: {shape}"
    )
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []

    for attempt in range(2):
        try:
            from openai import OpenAI

            prompt = user_prompt
            if attempt:
                prompt += "\nYour previous response was invalid. Return the corrected JSON array only."
            client = OpenAI(api_key=api_key)
            response = create_chat_completion(
                client,
                model=model,
                reasoning_effort=reasoning_effort,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            reply = response.choices[0].message.content or ""
            result = json.loads(_strip_code_fences(reply))
            if kind == "flashcards":
                if not isinstance(result, list) or len(result) != n:
                    continue
                if not all(
                    isinstance(item, dict)
                    and isinstance(item.get("front"), str)
                    and item["front"].strip()
                    and isinstance(item.get("back"), str)
                    and item["back"].strip()
                    for item in result
                ):
                    continue
                return [
                    {"front": item["front"].strip(), "back": item["back"].strip()}
                    for item in result
                ]

            if not isinstance(result, list) or len(result) != n:
                continue
            if not all(
                isinstance(item, dict)
                and isinstance(item.get("question"), str)
                and item["question"].strip()
                and isinstance(item.get("options"), list)
                and len(item["options"]) == 4
                and all(isinstance(option, str) for option in item["options"])
                and isinstance(item.get("answer_index"), int)
                and not isinstance(item.get("answer_index"), bool)
                and 0 <= item["answer_index"] <= 3
                and isinstance(item.get("explanation"), str)
                and item["explanation"].strip()
                for item in result
            ):
                continue
            return [
                {
                    "question": item["question"].strip(),
                    "options": [option.strip() for option in item["options"]],
                    "answer_index": item["answer_index"],
                    "explanation": item["explanation"].strip(),
                }
                for item in result
            ]
        except Exception:
            if attempt == 1:
                return []
    return []


def make_flashcards(
    text, n=5, model=MODEL, reasoning_effort=REASONING_EFFORT, api_key=None
):
    """Return up to n validated front/back flashcards based on text."""
    return _generate_list(
        text, n, model, "flashcards", reasoning_effort, api_key
    )


def make_quiz(
    text, n=5, model=MODEL, reasoning_effort=REASONING_EFFORT, api_key=None
):
    """Return up to n validated four-option questions based on text."""
    return _generate_list(text, n, model, "quiz", reasoning_effort, api_key)


if __name__ == "__main__":
    sample = (
        "Plants use sunlight to make food through photosynthesis. "
        "Chlorophyll in leaves captures light energy. "
        "Roots absorb water from the soil. "
        "Leaves take in carbon dioxide from the air. "
        "The process produces glucose and releases oxygen."
    )
    print("Flashcards:")
    print(make_flashcards(sample, n=3))
    print("Quiz:")
    print(make_quiz(sample, n=3))
