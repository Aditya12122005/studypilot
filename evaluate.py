"""Evaluate semantic retrieval against sample solar system notes."""

import argparse
from pathlib import Path

from modules.loader import clean_text, split_into_chunks
from modules.search import build_index, hybrid_search, search


TEST_CASES = [
    {
        "question": "What is the Sun mostly made of?",
        "expected": "hydrogen and helium",
    },
    {
        "question": "Which planet is closest to the Sun?",
        "expected": "closest planet to the Sun",
    },
    {
        "question": "What kind of atmosphere does Venus have?",
        "expected": "carbon dioxide atmosphere",
    },
    {
        "question": "What allows liquid water on Earth's surface?",
        "expected": "liquid water",
    },
    {
        "question": "Why does Mars look red?",
        "expected": "iron oxide",
    },
    {
        "question": "Which planet is the largest?",
        "expected": "largest planet",
    },
    {
        "question": "What are Saturn's rings mostly made of?",
        "expected": "pieces of ice",
    },
    {
        "question": "How does Uranus rotate?",
        "expected": "rotates on its side",
    },
    {
        "question": "What is notable about Neptune's winds?",
        "expected": "strongest winds",
    },
    {
        "question": "Where is Pluto located?",
        "expected": "Kuiper Belt",
    },
    {
        "question": "What substances dominate the luminous core of our neighborhood, and what invisible pull holds the surrounding worlds in place?",
        "expected": "hydrogen and helium",
    },
    {
        "question": "Which near-star body endures extreme hot-cold contrasts between its illuminated and dark halves, and how scant is its vapor layer?",
        "expected": "very little atmosphere",
    },
    {
        "question": "What lets our inhabited sphere retain H2O without freezing, and which airborne element is vital to respiration?",
        "expected": "contains oxygen",
    },
    {
        "question": "What mineral gives the fourth orb its scarlet appearance, and how many companions accompany it?",
        "expected": "iron oxide",
    },
    {
        "question": "Which massive orb features a long-lived crimson vortex observed across many generations?",
        "expected": "huge storm that has lasted for centuries",
    },
    {
        "question": "What frozen material makes up most of the luminous bands around the sixth orb?",
        "expected": "pieces of ice",
    },
    {
        "question": "Which remote globe is tipped nearly horizontal, and what vapor gives it a cyan tint?",
        "expected": "rotates on its side",
    },
    {
        "question": "Which far-off orbital zone contains the diminutive icy world?",
        "expected": "Kuiper Belt",
    },
]

OFF_TOPIC_QUESTIONS = [
    "Who won the 1998 cricket world cup?",
    "What is the capital city of Brazil?",
    "How do I bake sourdough bread?",
    "Who wrote the play Hamlet?",
    "How many moons does the Sun have?",
    "What is the temperature on Jupiter's largest moon?",
    "How many Earth years does Saturn take to orbit the Sun?",
    "What is the exact distance between Mars and the Moon?",
]

BIOLOGY_TEST_CASES = [
    {
        "question": "What boundary controls which substances can enter or leave a cell?",
        "expected": "controls what enters and leaves the cell",
        "kind": "paraphrase",
    },
    {
        "question": "Where is most cellular genetic information kept, and what molecule stores it?",
        "expected": "nucleus stores DNA",
        "kind": "two-fact",
    },
    {
        "question": "Which cell structure makes usable energy from glucose, and what gas does it use?",
        "expected": "mitochondria release energy from glucose using oxygen",
        "kind": "two-fact",
    },
    {
        "question": "How do plants turn light into food, and which two inputs do they need?",
        "expected": "uses carbon dioxide and water to make glucose and oxygen",
        "kind": "paraphrase and two-fact",
    },
    {
        "question": "Which plant parts take up water, and what leaf structures exchange gases?",
        "expected": "roots absorb water and minerals from soil, while stomata",
        "kind": "two-fact",
    },
    {
        "question": "Which plant tissues move water upward and distribute sugars from leaves?",
        "expected": "xylem carries water and minerals upward, while phloem distributes sugars",
        "kind": "two-fact",
    },
    {
        "question": "How can enzymes accelerate reactions without being used up?",
        "expected": "lowering activation energy and are not consumed",
        "kind": "paraphrase",
    },
    {
        "question": "During aerobic respiration, what do cells use, and what waste products form?",
        "expected": "cells use glucose and oxygen to release energy, producing carbon dioxide and water",
        "kind": "two-fact",
    },
    {
        "question": "Relative to the heart, which way do arteries and veins carry blood?",
        "expected": "arteries carry blood away from the heart, while veins return blood to the heart",
        "kind": "paraphrase and two-fact",
    },
    {
        "question": "Which blood cells transport oxygen, and which ones attack pathogens?",
        "expected": "red blood cells use hemoglobin to carry oxygen, while white blood cells attack pathogens",
        "kind": "two-fact",
    },
    {
        "question": "How do vaccines prepare immune defenses, and what helps the body respond faster later?",
        "expected": "vaccines train the immune system to recognize a pathogen, and memory cells support a faster response",
        "kind": "paraphrase and two-fact",
    },
    {
        "question": "What do decomposers do with dead material in an ecosystem?",
        "expected": "decomposers break down dead material and return nutrients to the environment",
        "kind": "paraphrase",
    },
]

BIOLOGY_OFF_TOPIC_QUESTIONS = [
    "Which antibiotic and dose should I take for strep throat?",
    "What percentage of participants improved in the 2024 Stanford microbiome trial?",
    "Which gut bacterium produces the most serotonin in adults?",
    "What is the half-life of carbon-14 in years?",
    "What is the recommended daily vitamin D intake for a ten-year-old?",
    "How many chromosomes does a horse have?",
    "What is the Latin binomial name of the blue whale?",
    "How can I culture E. coli at home for a science project?",
]

EVALUATION_SETS = [
    {
        "notes_file": "sample_notes.txt",
        "test_cases": TEST_CASES,
        "off_topic_questions": OFF_TOPIC_QUESTIONS,
    },
    {
        "notes_file": "notes_biology.txt",
        "test_cases": BIOLOGY_TEST_CASES,
        "off_topic_questions": BIOLOGY_OFF_TOPIC_QUESTIONS,
    },
]


def _retrieval_metrics(index, retrieve, test_cases):
    reciprocal_ranks = []
    hit_at_one = 0
    hit_at_three = 0
    for test_case in test_cases:
        results = retrieve(index, test_case["question"], top_k=3)
        expected = test_case["expected"].casefold()
        matching_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if expected in result["text"].casefold()
        ]
        first_match = matching_ranks[0] if matching_ranks else None
        hit_at_one += first_match == 1
        hit_at_three += first_match is not None
        reciprocal_ranks.append(1.0 / first_match if first_match else 0.0)
    total = len(test_cases)
    return hit_at_one, hit_at_three, sum(reciprocal_ranks) / total if total else 0.0


def _print_comparison(index, notes_file, test_cases):
    methods = [("Semantic", "n/a", search)]
    methods.extend(
        ("Hybrid", f"{alpha:.1f}", lambda idx, query, top_k, a=alpha: hybrid_search(
            idx, query, top_k=top_k, alpha=a
        ))
        for alpha in (0.5, 0.7, 0.9)
    )
    print(f"\n{notes_file}: retrieval comparison ({len(test_cases)} answerable questions)")
    print("Method   | Alpha | Hit@1 | Hit@3 | MRR")
    print("---------+-------+-------+-------+------")
    for method, alpha, retrieve in methods:
        hit_at_one, hit_at_three, mrr = _retrieval_metrics(
            index, retrieve, test_cases
        )
        print(
            f"{method:<8} | {alpha:>5} | "
            f"{hit_at_one:>2}/{len(test_cases):<2}  | "
            f"{hit_at_three:>2}/{len(test_cases):<2}  | {mrr:.4f}"
        )


def _evaluate_file(index, test_cases, off_topic_questions):
    answerable_rows = []
    reciprocal_ranks = []
    for test_case in test_cases:
        results = search(index, test_case["question"], top_k=3)
        expected = test_case["expected"].casefold()
        matching_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if expected in result["text"].casefold()
        ]
        first_match = matching_ranks[0] if matching_ranks else None
        hit_at_one = first_match == 1
        hit_at_three = first_match is not None
        reciprocal_ranks.append(1.0 / first_match if first_match else 0.0)
        top_score = results[0]["score"] if results else 0.0
        answerable_rows.append(
            (test_case["question"], hit_at_one, hit_at_three, top_score)
        )

    print(f"Answerable question retrieval ({len(test_cases)} questions)")
    question_width = max(
        len("Question"), *(len(row[0]) for row in answerable_rows)
    )
    print(
        f"{'Question':<{question_width}} | {'Hit@1':5} | "
        f"{'Hit@3':5} | Top score"
    )
    print(f"{'-' * question_width}-+-------+-------+----------")
    for question, hit_at_one, hit_at_three, top_score in answerable_rows:
        print(
            f"{question:<{question_width}} | {str(hit_at_one):5} | "
            f"{str(hit_at_three):5} | {top_score:.3f}"
        )

    hit_at_one_count = sum(row[1] for row in answerable_rows)
    hit_at_three_count = sum(row[2] for row in answerable_rows)
    mrr = sum(reciprocal_ranks) / len(test_cases) if test_cases else 0.0
    print(f"Hit@1: {hit_at_one_count}/{len(test_cases)}")
    print(f"Hit@3: {hit_at_three_count}/{len(test_cases)}")
    print(f"MRR: {mrr:.4f}")

    off_topic_rows = []
    print("Off-topic question scores")
    for question in off_topic_questions:
        results = search(index, question, top_k=3)
        top_score = results[0]["score"] if results else 0.0
        off_topic_rows.append((question, top_score))
        print(f"{question} | top score: {top_score:.3f}")

    answerable_scores = [row[3] for row in answerable_rows]
    off_topic_scores = [row[1] for row in off_topic_rows]
    print(
        "Lowest answerable top score: "
        f"{min(answerable_scores) if answerable_scores else 0.0:.3f}"
    )
    print(
        "Highest off-topic top score: "
        f"{max(off_topic_scores) if off_topic_scores else 0.0:.3f}"
    )

    total_questions = len(answerable_scores) + len(off_topic_scores)
    print("Threshold sweep")
    print("Min score | Wrongly rejected | Wrongly accepted | Accuracy")
    print("----------+------------------+------------------+---------")
    for threshold_step in range(30, 66, 5):
        min_score = threshold_step / 100.0
        wrongly_rejected = sum(score < min_score for score in answerable_scores)
        wrongly_accepted = sum(score >= min_score for score in off_topic_scores)
        correct = total_questions - wrongly_rejected - wrongly_accepted
        accuracy = correct / total_questions if total_questions else 0.0
        print(
            f"{min_score:9.2f} | {wrongly_rejected:16} | "
            f"{wrongly_accepted:16} | {accuracy:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-hybrid",
        action="store_true",
        help="compare semantic retrieval with hybrid alpha values 0.5, 0.7, and 0.9",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    for evaluation_set in EVALUATION_SETS:
        notes_file = evaluation_set["notes_file"]
        text = (project_root / notes_file).read_text(encoding="utf-8")
        cleaned_text = clean_text(text)
        chunks = split_into_chunks(cleaned_text, chunk_size=300, overlap=100)
        index = build_index(chunks)

        print(f"\n=== Results for {notes_file} ===")
        if args.compare_hybrid:
            _print_comparison(index, notes_file, evaluation_set["test_cases"])
        else:
            _evaluate_file(
                index,
                evaluation_set["test_cases"],
                evaluation_set["off_topic_questions"],
            )


if __name__ == "__main__":
    main()
