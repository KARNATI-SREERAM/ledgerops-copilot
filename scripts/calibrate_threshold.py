# Picks guardrails.MIN_SCORE from data instead of guessing. Prints the top cosine score of the
# retriever for on-topic and off-topic CALIBRATION questions. These are deliberately NOT the eval
# questions: tuning a threshold on the test set would make the eval score look better than it is.
# Run (needs Ollama): .venv/bin/python scripts/calibrate_threshold.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from copilot.llm import OllamaLLM  # noqa: E402
from copilot.retriever import Retriever  # noqa: E402

ON_TOPIC = [
    "Which port does the ledger API listen on?",
    "How are opening balances recorded?",
    "What does the Kafka consumer do with duplicate messages?",
    "Which features does the fraud model use?",
    "What happens when the connection pool is exhausted?",
    "How is the statement's running balance computed?",
    "What does the ledger return for an unknown account?",
    "Why is money stored as integer paise?",
]
OFF_TOPIC = [
    "What is the capital of Japan?",
    "Who won the cricket world cup in 2011?",
    "Write a poem about the monsoon.",
    "What is the boiling point of water in Kelvin?",
    "How do I bake sourdough bread?",
    "What is the current RBI repo rate?",
    "Translate 'good morning' into French.",
    "What's the best smartphone to buy this year?",
]


def main():
    retriever = Retriever(OllamaLLM())
    for label, questions in (("ON-TOPIC", ON_TOPIC), ("OFF-TOPIC", OFF_TOPIC)):
        scores = []
        for q in questions:
            top = retriever.search(q, k=1)[0]
            scores.append(top["score"])
            print(f"{label:9} {top['score']:.3f}  {q}  ->  {top['source']}")
        print(f"{label:9} min={min(scores):.3f} max={max(scores):.3f}\n")


if __name__ == "__main__":
    main()
