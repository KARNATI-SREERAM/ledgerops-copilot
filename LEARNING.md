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

### Phase 2: LangGraph agent with read-only ledger tools

**What was built**
- project2: a new read-only `GET /stats` (the 3 invariants, `healthy`, counts). `ledger.stats()` is now shared by the endpoint and
  `scripts/check_ledger.py`. A new `tests/test_stats.py` was added, and all 59 project2 tests pass. The commit was pushed and project2's CI is green.
- `copilot/llm.py`: the one LLM interface. `OllamaLLM` (generate + embed over HTTP, real token counts from Ollama) and `FakeLLM`
  (scripted by the `TASK:` line every prompt starts with). `get_llm()` picks one from `COPILOT_FAKE_LLM`.
- `copilot/retriever.py`: FAISS `IndexFlatIP` over normalized vectors (inner product = cosine). It reuses `chunk_markdown()` from Phase 1,
  and embeddings are cached in `index/`, keyed by a hash of the docs + the embedding model.
- `copilot/tools.py`: `get_health`, `get_stats`, `get_account`, `get_transfer`, `get_statement`. The only HTTP helper is `_get()`.
  Every `*_paise` field gets a `*_rupees` twin computed in Python.
- `copilot/agent.py`: graph `plan → (retrieve | call_tool) → plan … → answer → verify`. `MAX_STEPS = 5`, a repeated-action loop breaker,
  and `recursion_limit=25`. `verify` maps `[n]` citations back to chunks, drops invented numbers, and asks the model a one-word grounding question.
- `python -m copilot "question"` prints the full result state as JSON.
- Tests: 24 new ones (agent routing, step cap, loop breaker, invalid plan JSON, unknown tool, invented citations, verifier "no",
  tools only send GET, ₹ formatting, errors as data, retriever ranking + cache).

**What went wrong and was fixed (real debugging)**
- The verifier said "no" to a correct answer ("The balance of account 2 is ₹999,493.50 [T1]"). Two causes:
  the prompt showed an empty `Doc context: (none)` section and never explained what `[T1]` means. I tried two prompt
  versions on one correct and one deliberately wrong answer. Both judged correctly once the evidence was shown alone and the markers were explained.
  `json.dumps` was also escaping ₹ as `\u20b9` (now `ensure_ascii=False`).
- The 7B planner often proposes the *same* retrieval or tool call again instead of "answer". The loop breaker turns that into "answer",
  which is why most questions take 2 plan steps.

**Key concepts**
- **Agent vs chain.** A chain is a fixed sequence (Phase 1: retrieve → prompt → answer). Here the model's JSON decision picks the next node
  (`add_conditional_edges("plan", ...)`), so the path depends on the question.
- **LangGraph state and reducers.** Nodes return only the keys they change. `Annotated[list, operator.add]` makes `tool_calls`, `events`,
  `llm_calls` and `timings` *append* across loop iterations instead of being overwritten.
- **Tool calling with untrusted output.** The plan is validated by pydantic (`Plan`), the tool name by an allowlist, and the arguments by `_positive_int`.
  Each failure has a safe default instead of a crash.
- **Don't let the model do arithmetic.** Rupee strings are computed in code, and the prompt says to copy them.

**Break it to learn it**
1. Set `MAX_STEPS = 1`. Every question now answers after a single decision. A question needing a tool still works
   (plan → tool → forced answer), but one needing a tool *and* docs loses the second source.
2. Delete the `_already_done` check in `plan()`. Ask "What is the balance of account 2?" and count the `get_account` calls in `tool_calls`:
   the 7B planner calls it again until `MAX_STEPS` stops it, making the request several seconds slower.
3. Change `ensure_ascii=False` back to the default in `_context()`. Ask for an account balance: the verifier sees `\u20b9…` but the answer says `₹…`,
   and it starts refusing correct answers.
