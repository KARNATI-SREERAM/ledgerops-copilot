# HTTP API for the copilot: POST /ask {"question": "..."} returns a validated CopilotAnswer.
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


@app.get("/health")
def health():
    cp = app.state.copilot
    return {"status": "ok", "model": cp.llm.model, "embed_model": cp.llm.embed_model,
            "chunks_indexed": len(cp.retriever.chunks)}
