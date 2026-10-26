# Ask the copilot one question from the terminal and print the full result as JSON.
# Run: .venv/bin/python -m copilot "What is the balance of account 2?"
# (COPILOT_FAKE_LLM=1 uses the fake model; by default it uses Ollama on localhost.)

import json
import sys

from copilot.agent import Copilot
from copilot.llm import get_llm

if __name__ == "__main__":
    result = Copilot(get_llm()).ask(" ".join(sys.argv[1:]) or "Is the ledger healthy?")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
