# HTTP API for the copilot: POST /ask {"question": "..."} returns a validated CopilotAnswer.
# POST /ask/details returns the same answer plus what happened inside (for the demo UI): cited chunks
# with their retrieval scores, tool inputs and outputs, the guardrail decision, latency and tokens.
# Every request is also traced as one JSON line in traces/requests.jsonl (copilot/trace.py).
# Run locally: .venv/bin/uvicorn api:app --port 8002      (in compose: the `api` service, localhost:8002)

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from copilot.agent import Copilot
from copilot.llm import get_llm
from copilot.schemas import CopilotAnswer

TRACE_FILE = Path(os.getenv("TRACE_FILE", Path(__file__).parent / "traces" / "requests.jsonl"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index once at startup, not per request. Tests put their own copilot here first.
    if getattr(app.state, "copilot", None) is None:
        app.state.copilot = Copilot(get_llm(), trace_path=TRACE_FILE)
    yield


app = FastAPI(title="LedgerOps Copilot", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)  # a bound keeps prompts (and cost) bounded


@app.post("/ask", response_model=CopilotAnswer)
def ask(body: AskRequest):
    # A plain `def`: FastAPI runs it in a thread pool, so a slow model call doesn't block other requests.
    return app.state.copilot.ask(body.question)["output"]


# Refusals decided BEFORE the model runs (by regex on the question) are "blocked"; refusals decided
# later (weak retrieval, verifier said no...) are "refused". The UI shows the difference.
BLOCK_REASONS = {"prompt_injection", "write_blocked"}
WHY = {
    "prompt_injection": "The question matched a prompt-injection pattern, so it never reached the model.",
    "write_blocked": "The question asks to move money or change data. The copilot is read-only.",
    "low_retrieval_score": "No doc chunk was similar enough to the question, so there is nothing to ground an answer.",
    "model_said_no_support": "The model found no support for an answer in the retrieved docs or ledger data.",
    "no_valid_citation": "The draft answer cited no real doc chunk and no tool was used.",
    "verifier_unsupported": "The verifier call said the evidence does not support the draft answer.",
    "invalid_output": "The result did not fit the CopilotAnswer schema.",
}


def explain(final: dict, min_score: float) -> dict:
    """Turn the agent's final state into what the demo UI shows. Read-only: changes no decision."""
    out, events = final["output"], final.get("events", [])
    by_id = {c["id"]: c for c in final.get("chunks", [])}
    citations = [{"source": by_id[i]["source"], "score": by_id[i]["score"], "text": by_id[i]["text"]}
                 for i in final.get("cited_chunk_ids", []) if i in by_id]
    if not out["refused"]:
        decision = "allowed"
        why = "Input passed the injection and write checks; the answer is grounded and the verifier said yes."
    else:
        decision = "blocked" if out["reason"] in BLOCK_REASONS else "refused"
        why = WHY.get(out["reason"], out["reason"])
    low = next((e for e in events if e["type"] == "low_retrieval_score"), None)
    if low:
        why += f" Best score {low['top_score']:.3f} < threshold {low['min_score']:.2f}."
    match = next((e["pattern"] for e in events if "pattern" in e), None)  # the regex that blocked it
    llm_calls = final.get("llm_calls", [])
    return {
        "decision": decision,
        "why": why,
        "matched_pattern": match,
        "dropped_chunks": [e["chunk_id"] for e in events if e["type"] == "injection_in_chunk"],
        "min_score": min_score,  # the threshold citation scores are compared with
        "citations": citations,
        "tool_calls": [{k: c[k] for k in ("name", "args", "ok", "result", "ms")} for c in final.get("tool_calls", [])],
        "latency_ms": final["latency_ms"],
        "llm_calls": len(llm_calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in llm_calls),
        "completion_tokens": sum(c["completion_tokens"] for c in llm_calls),
    }


@app.post("/ask/details")
def ask_details(body: AskRequest):
    cp = app.state.copilot
    final = cp.ask(body.question)  # same call as /ask: same answer, same trace line
    return {"output": final["output"], "details": explain(final, cp.min_score)}


@app.get("/health")
def health():
    cp = app.state.copilot
    return {"status": "ok", "model": cp.llm.model, "embed_model": cp.llm.embed_model,
            "chunks_indexed": len(cp.retriever.chunks)}
