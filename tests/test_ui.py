# Renders ui.py headlessly with Streamlit's AppTest and a fake API reply, so the demo page is
# checked in CI without a browser, the API or Ollama.

from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

REPLY = {
    "output": {"answer": "Retry with the same Idempotency-Key. [1]", "citations": ["runbook.md § Recovery steps"],
               "tools_used": ["get_stats"], "refused": False, "reason": None},
    "details": {"decision": "allowed", "why": "grounded", "matched_pattern": None, "dropped_chunks": [],
                "min_score": 0.66, "citations": [{"source": "runbook.md § Recovery steps", "score": 0.81,
                                                  "text": "Restart postgres."}],
                "tool_calls": [{"name": "get_stats", "args": {}, "ok": True, "result": {"ok": True}, "ms": 12.0}],
                "latency_ms": 1234.0, "llm_calls": 3, "prompt_tokens": 900, "completion_tokens": 40},
}


def test_example_button_shows_answer_citation_tool_and_metrics():
    response = MagicMock()
    response.json.return_value = REPLY
    with patch("httpx.post", return_value=response) as post:
        at = AppTest.from_file("ui.py").run()
        at.sidebar.button[0].click().run()  # the "Docs question" example
    assert post.call_args.args[0].endswith("/ask/details")
    assert any("same Idempotency-Key" in m.value for m in at.markdown)
    assert at.success[0].value.startswith("**✅ ALLOWED**")
    labels = [e.label for e in at.expander]
    assert "[1] runbook.md § Recovery steps  ·  score 0.810" in labels
    assert any(lbl.startswith("🔧 get_stats") for lbl in labels)
    assert [m.value for m in at.metric] == ["1.23 s", "3", "900", "40"]
