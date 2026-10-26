# Tests for the from-scratch RAG (rag_minimal.py). No Ollama: embed() and call_llm() are replaced
# by fakes, so these run anywhere (including CI) in milliseconds.

import numpy as np

import rag_minimal

VOCAB = ["postgres", "crash", "idempotency", "kafka", "fraud", "rate", "limit"]


def fake_embed(texts):
    """Bag-of-words vectors over a tiny vocabulary: enough to make retrieval testable."""
    m = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32) + 1e-3
    return m / np.linalg.norm(m, axis=1, keepdims=True)


DOC = """# Title
Intro text.

## Idempotency
Retries reuse the same key.

```bash
# this is a shell comment, not a heading
echo hi
```

## Empty parent
### Child
Child text.
"""


def test_chunks_keep_file_and_section_and_ignore_code_comments():
    chunks = rag_minimal.chunk_markdown(DOC, "project2/LEARNING.md")
    sources = [c["source"] for c in chunks]
    assert sources == ["project2/LEARNING.md § Title", "project2/LEARNING.md § Idempotency",
                       "project2/LEARNING.md § Empty parent > Child"]  # "Empty parent" has no text of its own
    assert "# this is a shell comment" in chunks[1]["text"]
    assert chunks[1]["id"] == "project2/LEARNING.md#idempotency-0"


def test_same_named_sections_get_unique_ids_and_their_parent_in_the_name():
    doc = "# Runbook\n## Crash\n### Recovery steps\nRestart postgres.\n## 429\n### Recovery steps\nWait.\n"
    chunks = rag_minimal.chunk_markdown(doc, "runbook.md")
    assert [c["source"] for c in chunks] == ["runbook.md § Crash > Recovery steps", "runbook.md § 429 > Recovery steps"]
    assert len({c["id"] for c in chunks}) == 2


def test_real_corpus_chunk_ids_are_unique():
    ids = [c["id"] for name in rag_minimal.DOCS
           for c in rag_minimal.chunk_markdown((rag_minimal.DOCS_DIR / name).read_text(encoding="utf-8"), name)]
    assert len(ids) == len(set(ids))


def test_long_sections_are_windowed_with_overlap():
    body = "".join(f"{i:04d} " for i in range(400))  # 2000 characters
    chunks = rag_minimal.chunk_markdown(f"## Big\n{body}", "x.md")
    assert len(chunks) == 3  # windows start at 0, 650, 1300 (size 800, overlap 150)
    first, second = chunks[0]["text"], chunks[1]["text"]
    # The last 150 characters of window 1 are the first 150 characters of window 2.
    assert first[-150:] == second[len("Big\n"):len("Big\n") + 150]
    assert chunks[-1]["text"].endswith(body.strip()[-10:])


def test_retrieve_ranks_the_matching_chunk_first(monkeypatch):
    monkeypatch.setattr(rag_minimal, "embed", fake_embed)
    chunks = [{"id": "a", "source": "a § A", "text": "kafka consumer groups"},
              {"id": "b", "source": "b § B", "text": "postgres crash recovery"},
              {"id": "c", "source": "c § C", "text": "fraud model recall"}]
    index = {"chunks": chunks, "vectors": fake_embed([c["text"] for c in chunks])}
    top = rag_minimal.retrieve(index, "what happens on a postgres crash?", k=2)
    assert top[0]["id"] == "b"
    assert top[0]["score"] > top[1]["score"]


def test_prompt_numbers_chunks_shows_sources_and_the_refusal_rule():
    chunks = [{"source": "runbook.md § Recovery steps", "text": "Restart postgres."}]
    prompt = rag_minimal.build_prompt("How to recover?", chunks)
    assert "[1] (source: runbook.md § Recovery steps)" in prompt
    assert "I don't know from the docs" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "Question: How to recover?" in prompt


def test_init_indexes_every_doc(monkeypatch):
    monkeypatch.setattr(rag_minimal, "embed", fake_embed)
    index = rag_minimal.init()
    files = {c["source"].split(" § ")[0] for c in index["chunks"]}
    assert files == set(rag_minimal.DOCS)
    assert index["vectors"].shape[0] == len(index["chunks"])
    assert np.allclose(np.linalg.norm(index["vectors"], axis=1), 1.0)  # unit vectors: dot = cosine
