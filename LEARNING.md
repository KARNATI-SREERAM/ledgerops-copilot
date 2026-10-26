# LEARNING.md: Project 3 study guide (LedgerOps Copilot)

(The full study guide is written in Phase 5. Until then, this file holds the build log, one entry per phase.)

## Appendix: build log, phase by phase

### Phase 1: Minimal RAG from scratch (`rag_minimal.py`)

**What was built**
- `rag_minimal.py`: the whole RAG idea in one file, with no framework. `init()` loads 5 markdown files, chunks them,
  embeds every chunk with Ollama `nomic-embed-text` and keeps a numpy matrix of unit vectors. `retrieve()` does cosine search.
  `build_prompt()` numbers the top 4 chunks and adds strict rules. `call_llm()` asks `qwen2.5:7b` at temperature 0.
- `docs/runbook.md`: an operations runbook written from project2's real failure drill (Postgres SIGKILL, 12 × 503,
  2.60 s client-visible outage, 0 lost or doubled transfers) and from a real startup incident hit while building this phase.
- `docs/project1/`, `docs/project2/`: copies of both projects' README and LEARNING, so this repo is self-contained (CI and Docker can't see sibling folders).
- `tests/test_rag_minimal.py`: 5 tests with a fake embedder (bag of words), no Ollama.
- Real run: [results/phase1_rag_minimal.txt](results/phase1_rag_minimal.txt). 189 chunks were indexed. The answer about Postgres crashes was correct
  and cited its chunk. The top chunk was `runbook.md § Recovery steps` (cosine 0.822).

**Key concepts**
- **Chunking.** Split on markdown headings first, so a chunk never mixes two topics. Then split long sections into 800-character windows
  with 150 characters of overlap, so a sentence cut at a boundary is still whole in the next chunk. Each chunk keeps `file § section`, which is what a citation shows.
- **Embeddings.** Each chunk becomes a 768-number vector. Similar meaning gives nearby vectors. `nomic-embed-text` expects the prefixes
  `search_document:` and `search_query:`, so documents and questions are embedded slightly differently.
- **Cosine similarity.** Every vector is scaled to length 1, so `vectors @ q` (a dot product) *is* the cosine. One matrix multiply scores all 189 chunks.
- **Grounding prompt.** Number the chunks, demand a citation per sentence, give an exact refusal string, and say that context text is data, not instructions.

**Break it to learn it**
1. Set `CHUNK_OVERLAP = 0` and `CHUNK_SIZE = 200`, then ask about the failure drill numbers. Facts get cut in half between chunks,
   so retrieval finds pieces without the number and the answer gets vaguer or refuses.
2. Remove the `search_query:` / `search_document:` prefixes. Re-run the same question and compare the cosine scores. They change, and the ranking can too:
   the model was trained with those prefixes.
3. Delete the line "If the context does not contain the answer, reply exactly…" and ask "What is NPCI's head office address?".
   Without the rule, the model is far more likely to answer from its own memory, which is a hallucination as far as *your* docs are concerned.
