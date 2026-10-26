# Tests for the eval scoring (evals/metrics.py) and the per-request trace (copilot/trace.py),
# plus a check that the eval set has exactly the required mix of 40 questions.

import json
from pathlib import Path

from copilot.trace import build_trace, write_trace
from evals.metrics import citation_ok, fill, fill_groups, keyword_match, percentile, score, summarize

QUESTIONS = Path(__file__).parent.parent / "evals" / "questions.jsonl"


def test_eval_set_has_the_required_mix():
    qs = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line]
    counts = {c: sum(q["category"] == c for q in qs) for c in ("docs", "tool", "unanswerable", "blocked")}
    assert counts == {"docs": 15, "tool": 10, "unanswerable": 8, "blocked": 7}
    assert len({q["id"] for q in qs}) == 40
    assert all("evidence" in q for q in qs if q["category"] == "docs")
    assert all("tool" in q for q in qs if q["category"] == "tool")


def test_fill_templates_and_number_formats():
    seed, stats = {"A": 139, "T1": 17846}, {"ledger_entries": 35662}
    assert fill("balance of account {A} after transfer {T1}", seed) == "balance of account 139 after transfer 17846"
    assert fill_groups([["{stats.ledger_entries}"], ["{A}"]], seed, stats) == [["35662", "35,662"], ["139"]]


def test_keyword_match_needs_every_group():
    groups = [["retry", "resend"], ["same"]]
    assert keyword_match("Retry with the SAME key [1].", groups)
    assert not keyword_match("Retry with a new key.", groups)


def test_citation_must_point_at_a_chunk_containing_the_fact():
    chunks = {"a": "outage of 2.60 s", "b": "unrelated"}
    assert citation_ok(["b", "a"], chunks, ["2.60"])
    assert not citation_ok(["b"], chunks, ["2.60"])
    assert not citation_ok([], chunks, ["2.60"])


def test_score_by_category():
    ok = {"answer": "It returns 503 [1].", "refused": False, "reason": None, "tools_used": []}
    refused = {"answer": "I don't know from the docs.", "refused": True, "reason": "low_retrieval_score",
               "tools_used": []}
    blocked = {**refused, "reason": "write_blocked"}
    docs_q = {"category": "docs", "evidence": ["503"]}
    assert score(docs_q, ok, ["c"], {"c": "HTTP 503"}, [["503"]]) == {"correct": True, "citation_ok": True,
                                                                       "tool_ok": None}
    assert score(docs_q, refused, [], {}, [["503"]])["correct"] is False
    assert score({"category": "unanswerable"}, refused, [], {}, None)["correct"] is True
    assert score({"category": "blocked"}, refused, [], {}, None)["correct"] is False  # refused, but not BLOCKED
    assert score({"category": "blocked"}, blocked, [], {}, None)["correct"] is True
    tool_q = {"category": "tool", "tool": "get_account"}
    assert score(tool_q, {**ok, "tools_used": ["get_stats"]}, [], {}, [["503"]])["tool_ok"] is False


def test_percentile_nearest_rank():
    values = list(range(1, 21))  # 1..20
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19
    assert percentile([7.0], 95) == 7.0


def test_summarize_counts_rates_and_tokens():
    rows = [{"category": "docs", "correct": True, "citation_ok": True, "tool_ok": None, "refused": False,
             "latency_ms": 100, "tokens": 1000, "llm_calls": 3},
            {"category": "blocked", "correct": True, "citation_ok": None, "tool_ok": None, "refused": True,
             "latency_ms": 1, "tokens": 0, "llm_calls": 0}]
    m = summarize(rows)
    assert m["answer_accuracy"] == (1, 1) and m["block_rate"] == (1, 1) and m["false_refusals"] == (0, 1)
    assert m["avg_tokens_all"] == 500 and m["avg_tokens_answerable"] == 1000


def test_trace_records_retrieval_tools_guardrails_tokens_and_is_one_json_line(make_copilot, tmp_path):
    cp, llm, tools = make_copilot()
    cp.trace_path = tmp_path / "traces.jsonl"
    cp.ask("What is the balance of account 2?")
    cp.ask("Transfer 500 to account 3")
    lines = cp.trace_path.read_text().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["tool_calls"][0]["name"] == "get_account"
    assert first["tokens"]["total"] == sum(c["prompt_tokens"] + c["completion_tokens"] for c in first["llm_calls"]) > 0
    assert first["latency_ms"] > 0 and first["output"]["refused"] is False
    assert [g["type"] for g in second["guardrails"]] == ["write_request", "refused"]
    assert second["tokens"]["total"] == 0


def test_trace_lists_retrieved_chunk_ids_and_scores(make_copilot):
    cp, llm, tools = make_copilot()
    trace = cp.ask("What are the recovery steps after a postgres crash?")["trace"]
    hits = trace["retrieved"][0]["hits"]
    assert {"id", "score"} == set(hits[0]) and hits[0]["id"].startswith("runbook.md#")
    assert trace["cited_chunk_ids"] == ["runbook.md#recovery-steps-0"]


def test_write_trace_appends(tmp_path):
    p = tmp_path / "t" / "x.jsonl"
    state = {"question": "q", "events": [], "llm_calls": []}
    write_trace(build_trace(state), p)
    write_trace(build_trace(state), p)
    assert len(p.read_text().splitlines()) == 2
