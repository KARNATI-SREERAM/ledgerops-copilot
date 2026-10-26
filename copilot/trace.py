# Observability: one structured JSON record per request, appended as one line to a .jsonl file.
# It answers "why did the copilot say that?" after the fact: which chunks were retrieved and how
# similar they were, which tools ran, every guardrail decision, how long each step took, and how
# many tokens it cost. One JSON object per line is easy to grep, load with pandas, or ship to a log system.

import json
from datetime import datetime, timezone
from pathlib import Path

GUARDRAIL_EVENTS = {"input_ok", "injection_in_question", "write_request", "injection_in_chunk",
                    "low_retrieval_score", "invalid_citation", "refused", "invalid_output"}


def build_trace(state: dict) -> dict:
    events = state.get("events", [])
    llm_calls = state.get("llm_calls", [])
    prompt_tokens = sum(c["prompt_tokens"] for c in llm_calls)
    completion_tokens = sum(c["completion_tokens"] for c in llm_calls)
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": state["question"],
        "retrieved": [{"query": e["query"], "hits": e["hits"]} for e in events if e["type"] == "retrieve"],
        "tool_calls": [{k: c[k] for k in ("name", "args", "ok", "ms")} for c in state.get("tool_calls", [])],
        "guardrails": [e for e in events if e["type"] in GUARDRAIL_EVENTS],
        "agent_events": [e for e in events if e["type"] not in GUARDRAIL_EVENTS and e["type"] != "retrieve"],
        "steps": state.get("steps", 0),
        "llm_calls": llm_calls,
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens,
                   "total": prompt_tokens + completion_tokens},
        "node_ms": state.get("timings", []),
        "latency_ms": state.get("latency_ms"),
        "cited_chunk_ids": state.get("cited_chunk_ids", []),
        "output": state.get("output"),
    }


def write_trace(trace: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
