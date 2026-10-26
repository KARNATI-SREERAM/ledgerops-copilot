# The ONE interface between the copilot and a language model. Everything else calls
# llm.generate(prompt) and llm.embed(texts), and never talks to Ollama directly.
# Why: tests and CI swap in FakeLLM (deterministic, instant, no Ollama), and moving to another
# model or provider later means changing this file only.
#   OllamaLLM : the real thing, qwen2.5:7b + nomic-embed-text over Ollama's HTTP API
#   FakeLLM   : scripted replies chosen by the "TASK:" line every prompt starts with
#   get_llm() : picks one from the COPILOT_FAKE_LLM environment variable

import json
import os
import re
import zlib
from dataclasses import dataclass

import httpx
import numpy as np


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int      # tokens the model read (Ollama: prompt_eval_count)
    completion_tokens: int  # tokens the model wrote (Ollama: eval_count)


class OllamaLLM:
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.http = httpx.Client(base_url=self.url, timeout=300)

    def generate(self, prompt: str, json_mode: bool = False, max_tokens: int = 400) -> LLMResult:
        body = {"model": self.model, "prompt": prompt, "stream": False,
                # temperature 0: same question -> same answer, which makes evals repeatable.
                # num_ctx 4096: room for 4 chunks + tool output; a bigger window costs more RAM.
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}}
        if json_mode:
            body["format"] = "json"  # Ollama constrains the output to valid JSON
        r = self.http.post("/api/generate", json=body)
        r.raise_for_status()
        d = r.json()
        return LLMResult(d["response"].strip(), d.get("prompt_eval_count", 0), d.get("eval_count", 0))

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), 32):
            r = self.http.post("/api/embed", json={"model": self.embed_model, "input": texts[i:i + 32]})
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
        return np.array(vectors, dtype=np.float32)


class FakeLLM:
    """Deterministic stand-in for tests and CI. It looks at the TASK line and the question,
    and replies the way a well-behaved model would. `replies` can override any task."""

    model = "fake"
    embed_model = "fake-hash-256"

    def __init__(self, replies: dict | None = None):
        self.replies = replies or {}  # e.g. {"plan": ['{"action": "answer"}'], "verify": ["no"]}
        self.prompts: list[str] = []  # every prompt received, so tests can inspect them

    def generate(self, prompt: str, json_mode: bool = False, max_tokens: int = 400) -> LLMResult:
        self.prompts.append(prompt)
        task = prompt.split("\n", 1)[0].removeprefix("TASK: ").strip()
        scripted = self.replies.get(task)
        text = scripted.pop(0) if scripted else getattr(self, f"_{task}")(prompt)
        return LLMResult(text, len(prompt.split()), len(text.split()))

    def embed(self, texts: list[str]) -> np.ndarray:
        # Hashed bag of words: texts sharing words get similar vectors. Good enough to test retrieval.
        m = np.zeros((len(texts), 256), dtype=np.float32)
        for row, t in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", t.lower()):
                m[row, zlib.crc32(word.encode()) % 256] += 1
        return m + 1e-6  # never an all-zero vector (it could not be normalized)

    # ---- default scripted behaviour, one method per TASK ----
    def _plan(self, prompt: str) -> str:
        q = _section(prompt, "User question:").lower()
        if "Observations so far: none" not in prompt:
            return json.dumps({"action": "answer"})
        acc, tr = re.search(r"account\s*#?(\d+)", q), re.search(r"transfer\s*#?(\d+)", q)
        if acc and "statement" in q:
            return json.dumps({"action": "tool", "tool": "get_statement",
                               "args": {"account_id": int(acc.group(1))}})
        if acc:
            return json.dumps({"action": "tool", "tool": "get_account",
                               "args": {"account_id": int(acc.group(1))}})
        if tr:
            return json.dumps({"action": "tool", "tool": "get_transfer",
                               "args": {"transfer_id": int(tr.group(1))}})
        if "healthy" in q or "health" in q or " up" in q:
            return json.dumps({"action": "tool", "tool": "get_health", "args": {}})
        if "how many" in q or "stats" in q:
            return json.dumps({"action": "tool", "tool": "get_stats", "args": {}})
        return json.dumps({"action": "retrieve"})

    def _answer(self, prompt: str) -> str:
        if "\n[T1] " in prompt:  # at the start of a line: real ledger data, not the rules text
            return f"The ledger returned: {_section(prompt, chr(10) + '[T1] ')[:200]} [T1]"
        if "\n[1] (source:" in prompt:
            return f"According to the docs: {_section(prompt, chr(10) + '[1] (source:')[:200]} [1]"
        return "I don't know from the docs"

    def _verify(self, prompt: str) -> str:
        return "yes"


def _section(prompt: str, marker: str) -> str:
    """The text after `marker` up to the next blank line (used by the fake to read its prompt)."""
    after = prompt.split(marker, 1)[1] if marker in prompt else ""
    return after.split("\n\n", 1)[0].strip()


def get_llm():
    return FakeLLM() if os.getenv("COPILOT_FAKE_LLM") == "1" else OllamaLLM()
