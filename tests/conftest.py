# Shared fixtures: a FakeLLM, a retriever over a tiny temporary corpus, and fake ledger tools.
# Nothing here needs Ollama or the ledger, which is why CI can run the whole suite.

import pytest

from copilot.agent import Copilot
from copilot.llm import FakeLLM
from copilot.retriever import Retriever

DOCS = {
    "runbook.md": "# Runbook\n## Recovery steps\nRestart postgres, then clients retry with the same Idempotency-Key.\n"
                  "## Rate limiting\nEach client gets a token bucket: burst 100, refill 50 requests per second.\n",
    "project2/LEARNING.md": "# Ledger\n## Double-entry\nEvery transfer writes one DEBIT and one CREDIT entry.\n",
}

FAKE_RESULTS = {
    "get_account": {"ok": True, "data": {"id": 2, "owner_name": "Asha", "balance_paise": 37500,
                                         "balance_rupees": "₹375.00"}},
    "get_health": {"ok": True, "data": {"status": "ok"}},
    "get_stats": {"ok": True, "data": {"healthy": True, "customer_accounts": 3}},
    "get_transfer": {"ok": True, "data": {"id": 9, "amount_paise": 12500, "amount_rupees": "₹125.00"}},
    "get_statement": {"ok": True, "data": {"account_id": 2, "entries": []}},
}


class FakeTools:
    """Records every call, returns canned results."""
    def __init__(self):
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, args))
        return FAKE_RESULTS.get(name, {"ok": False, "error": "unknown tool"})


@pytest.fixture
def corpus(tmp_path):
    for name, text in DOCS.items():
        (tmp_path / "docs" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs" / name).write_text(text)
    return tmp_path


@pytest.fixture
def make_copilot(corpus):
    """make_copilot(replies) -> (copilot, llm, tools). `replies` scripts the FakeLLM per TASK."""
    def make(replies=None):
        llm, tools = FakeLLM(replies), FakeTools()
        retriever = Retriever(llm, docs_dir=corpus / "docs", docs=list(DOCS), cache_dir=corpus / "index")
        return Copilot(llm, retriever=retriever, tool_fn=tools), llm, tools
    return make
