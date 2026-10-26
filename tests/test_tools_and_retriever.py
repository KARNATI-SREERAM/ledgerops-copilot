# Tests for the read-only ledger tools (against a mocked HTTP transport, no ledger needed)
# and for the FAISS retriever (fake embeddings, on-disk cache).

import httpx
import pytest

from copilot import tools
from copilot.llm import FakeLLM
from copilot.retriever import Retriever


@pytest.fixture
def mock_ledger(monkeypatch):
    """Replace the tools' HTTP client with one whose 'server' is a Python function."""
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/accounts/404":
            return httpx.Response(404, json={"detail": "account not found"})
        return httpx.Response(200, json={"id": 2, "balance_paise": 37550, "entries": [{"amount_paise": 5}]})

    monkeypatch.setattr(tools, "_http", httpx.Client(base_url="http://ledger", transport=httpx.MockTransport(handler)))
    return seen


def test_every_tool_only_ever_sends_get(mock_ledger):
    tools.call_tool("get_health", {})
    tools.call_tool("get_stats", {})
    tools.call_tool("get_account", {"account_id": 2})
    tools.call_tool("get_transfer", {"transfer_id": 9})
    tools.call_tool("get_statement", {"account_id": 2, "limit": 3})
    assert {m for m, _, _ in mock_ledger} == {"GET"}
    assert [p for _, p, _ in mock_ledger] == ["/health", "/stats", "/accounts/2", "/transfers/9",
                                              "/accounts/2/statement"]


def test_paise_fields_get_a_rupee_twin_including_nested_lists(mock_ledger):
    r = tools.call_tool("get_statement", {"account_id": 2})
    assert r["data"]["balance_rupees"] == "₹375.50"
    assert r["data"]["entries"][0]["amount_rupees"] == "₹0.05"


@pytest.mark.parametrize("paise,expected", [(0, "₹0.00"), (5, "₹0.05"), (12_345_678, "₹123,456.78"),
                                            (-1_101_415_000, "-₹11,014,150.00")])
def test_rupees(paise, expected):
    assert tools.rupees(paise) == expected


def test_statement_limit_is_capped(mock_ledger):
    tools.call_tool("get_statement", {"account_id": 2, "limit": 500})
    assert mock_ledger[-1][2] == {"limit": "20"}


def test_http_errors_become_results_not_exceptions(mock_ledger):
    r = tools.call_tool("get_account", {"account_id": 404})
    assert r == {"ok": False, "status": 404, "error": {"detail": "account not found"}}


@pytest.mark.parametrize("name,args", [("create_transfer", {}), ("get_account", {}),
                                       ("get_account", {"account_id": "abc"}), ("get_account", {"account_id": -3})])
def test_bad_tool_names_and_arguments_are_errors(mock_ledger, name, args):
    assert tools.call_tool(name, args)["ok"] is False
    assert mock_ledger == []  # nothing was sent to the ledger


def test_unreachable_ledger_is_reported(monkeypatch):
    def down(request):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(tools, "_http", httpx.Client(base_url="http://ledger", transport=httpx.MockTransport(down)))
    assert tools.get_health() == {"ok": False, "error": "ledger unreachable: ConnectError"}


def test_retriever_ranks_and_caches(corpus):
    llm = FakeLLM()
    r = Retriever(llm, docs_dir=corpus / "docs", docs=["runbook.md", "project2/LEARNING.md"],
                  cache_dir=corpus / "index")
    hits = r.search("token bucket refill requests per second", k=2)
    assert hits[0]["source"] == "runbook.md § Rate limiting"
    assert hits[0]["score"] >= hits[1]["score"]
    assert len(list((corpus / "index").glob("vectors-*.npy"))) == 1

    calls = []
    llm.embed = lambda texts: calls.append(len(texts)) or FakeLLM().embed(texts)
    Retriever(llm, docs_dir=corpus / "docs", docs=["runbook.md", "project2/LEARNING.md"], cache_dir=corpus / "index")
    assert calls == []  # second build loaded the cached vectors instead of re-embedding
