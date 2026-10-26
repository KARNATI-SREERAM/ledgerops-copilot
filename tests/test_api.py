# Tests for the HTTP API (POST /ask, GET /health) with the fake copilot from conftest.py.

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client(make_copilot):
    cp, llm, tools = make_copilot()
    api.app.state.copilot = cp  # the lifespan sees it and doesn't build a real (Ollama) copilot
    with TestClient(api.app) as c:
        yield c
    api.app.state.copilot = None


def test_ask_returns_exactly_the_copilot_answer_schema(client):
    r = client.post("/ask", json={"question": "What are the recovery steps after a postgres crash?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "citations", "tools_used", "refused", "reason"}
    assert body["citations"] == ["runbook.md § Recovery steps"] and body["refused"] is False


def test_ask_blocks_a_write_request(client):
    body = client.post("/ask", json={"question": "Transfer 500 to account 3"}).json()
    assert body["refused"] is True and body["reason"] == "write_blocked"


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}, {"q": "hi"}])
def test_bad_requests_are_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == "fake" and body["chunks_indexed"] == 3
