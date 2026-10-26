# Runs the 40-question eval against the REAL copilot (Ollama qwen2.5:7b + nomic-embed-text) and the
# REAL ledger, then writes:
#   results/eval_results.json   every question: answer, refusal, scores, latency, tokens
#   results/eval_traces.jsonl   the structured trace of every request (copilot/trace.py)
#   results/eval_run.txt        environment + the metrics table (the numbers quoted in README/LEARNING)
# Run (ledger up on :8001, Ollama running): .venv/bin/python evals/run_evals.py

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from copilot import tools  # noqa: E402
from copilot.agent import Copilot  # noqa: E402
from copilot.llm import OllamaLLM  # noqa: E402
from evals.metrics import fill, fill_groups, score, summarize  # noqa: E402
from scripts.seed_ledger import seed  # noqa: E402

RESULTS = ROOT / "results"


def sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return "?"


def environment(llm: OllamaLLM) -> list[str]:
    version = llm.http.get("/api/version").json().get("version", "?")
    dirty = " (plus uncommitted changes)" if sh("git", "status", "--porcelain", "--untracked-files=no") else ""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [f"date: {now}  git: {sh('git', 'rev-parse', '--short', 'HEAD')}{dirty}",
            f"machine: {sh('sysctl', '-n', 'machdep.cpu.brand_string') or platform.processor()}, "
            f"{int(sh('sysctl', '-n', 'hw.memsize') or 0) // 2**30} GB RAM, {platform.system()} {platform.release()}",
            f"ollama {version}: chat model {llm.model}, embeddings {llm.embed_model}; temperature 0, num_ctx 4096",
            f"ledger: {tools.LEDGER_URL}"]


def fmt(pair) -> str:
    hit, n = pair
    return f"{hit}/{n} = {100 * hit / n:.1f}%" if n else "n/a"


def table(m: dict) -> list[str]:
    return [
        "| Metric | Result |", "|---|---|",
        f"| Answer accuracy (keyword match), docs + tool questions | {fmt(m['answer_accuracy'])} |",
        f"| ... docs questions | {fmt(m['answer_accuracy_docs'])} |",
        f"| ... tool questions | {fmt(m['answer_accuracy_tool'])} |",
        f"| Citation accuracy (cited chunk contains the fact), docs questions | {fmt(m['citation_accuracy'])} |",
        f"| Right tool called, tool questions | {fmt(m['tool_selection'])} |",
        f"| Correct-refusal rate, unanswerable questions | {fmt(m['correct_refusal'])} |",
        f"| Block rate, injection + write attempts | {fmt(m['block_rate'])} |",
        f"| False refusals, answerable questions | {fmt(m['false_refusals'])} |",
        f"| Latency p50 / p95, all {m['n']} questions | "
        f"{m['latency_p50_ms'] / 1000:.2f} s / {m['latency_p95_ms'] / 1000:.2f} s |",
        "| Latency p50 by category | " + ", ".join(f"{c} {v / 1000:.2f} s"
                                           for c, v in m["latency_p50_ms_by_category"].items()) + " |",
        f"| Avg tokens per query (prompt + completion), all {m['n']} | {m['avg_tokens_all']:.0f} |",
        f"| Avg tokens per query, answerable questions only | {m['avg_tokens_answerable']:.0f} |",
        f"| Avg LLM calls per query | {m['avg_llm_calls']:.2f} |",
    ]


def main():
    questions = [json.loads(line) for line in (ROOT / "evals" / "questions.jsonl").read_text().splitlines() if line]
    llm = OllamaLLM()
    env = environment(llm)
    ids = seed()
    print("seeded:", ids)

    traces = RESULTS / "eval_traces.jsonl"
    traces.unlink(missing_ok=True)
    copilot = Copilot(llm, trace_path=traces)  # builds (or loads) the FAISS index before any timing
    chunk_text = {c["id"]: c["text"] for c in copilot.retriever.chunks}
    # Warm-up, NOT measured: the first call loads the models into memory (several seconds, once).
    llm.generate("Say OK.", max_tokens=2)
    llm.embed(["warm up"])

    rows = []
    started = time.perf_counter()
    for q in questions:
        stats = tools.get_stats().get("data") if "{stats." in json.dumps(q) else None  # live ground truth
        question = fill(q["question"], ids, stats)
        expected = fill_groups(q["expected"], ids, stats) if "expected" in q else None
        state = copilot.ask(question)
        out = state["output"]
        row = {"id": q["id"], "category": q["category"], "question": question, "expected": expected,
               **out, "cited_chunk_ids": state.get("cited_chunk_ids", []),
               **score(q, out, state.get("cited_chunk_ids", []), chunk_text, expected),
               "latency_ms": state["latency_ms"], "tokens": state["trace"]["tokens"]["total"],
               "llm_calls": len(state.get("llm_calls", [])), "steps": state.get("steps", 0)}
        rows.append(row)
        mark = {True: "PASS", False: "FAIL", None: "-"}[row["correct"]]
        print(f"{mark} {q['id']} {row['latency_ms'] / 1000:5.1f}s refused={out['refused']} {question[:70]}")
    total = time.perf_counter() - started

    m = summarize(rows)
    failures = [f"{r['id']} [{r['category']}] {r['question']}\n    -> refused={r['refused']} reason={r['reason']} "
                f"answer={r['answer'][:220]!r}" + (f"\n    expected={r['expected']}" if r["expected"] else "")
                for r in rows if r["correct"] is False or r["citation_ok"] is False or r["tool_ok"] is False]
    report = ["== Environment", *env, f"seeded ledger ids: {ids}",
              f"== Metrics ({len(rows)} questions, one run, {total:.0f} s total; warm-up call excluded)", *table(m),
              "", "== Questions that failed a check (answer, citation or tool)", *(failures or ["(none)"])]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval_results.json").write_text(json.dumps({"metrics": m, "rows": rows}, indent=2, ensure_ascii=False))
    (RESULTS / "eval_run.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
