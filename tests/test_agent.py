# Tests for the LangGraph agent: routing (docs vs tools), the step cap, the loop breaker,
# bad model output, and verification. The FakeLLM's replies are scripted per TASK where needed.

import json

from copilot.agent import MAX_STEPS, REFUSAL


def plan(**kw):
    return json.dumps(kw)


def test_docs_question_retrieves_then_answers_with_a_citation(make_copilot):
    cp, llm, tools = make_copilot()
    r = cp.ask("What are the recovery steps after a postgres crash?")
    assert not r["refused"]
    assert r["citations"] == ["runbook.md § Recovery steps"]
    assert tools.calls == []
    assert [t["node"] for t in r["timings"]] == ["plan", "retrieve", "plan", "answer", "verify"]


def test_live_data_question_calls_the_tool_with_the_right_argument(make_copilot):
    cp, llm, tools = make_copilot()
    r = cp.ask("What is the balance of account 2?")
    assert tools.calls == [("get_account", {"account_id": 2})]
    assert r["tools_used"] == ["get_account"]
    assert not r["refused"]
    assert "₹375.00" in r["answer"]


def test_step_cap_stops_a_model_that_never_wants_to_answer(make_copilot):
    # A planner that asks for a new retrieval forever (different query each time, so the loop breaker
    # does not fire). Only MAX_STEPS stops it.
    endless = [plan(action="retrieve", query=f"query {i}") for i in range(50)]
    cp, llm, tools = make_copilot({"plan": endless})
    r = cp.ask("loop forever")
    assert r["steps"] == MAX_STEPS + 1  # 5 real decisions, then the forced "answer"
    assert {"type": "step_cap"} in r["events"]
    assert sum(t["node"] == "retrieve" for t in r["timings"]) == MAX_STEPS
    assert "verify" in [t["node"] for t in r["timings"]]


def test_repeating_the_same_tool_call_is_turned_into_answer(make_copilot):
    same = plan(action="tool", tool="get_health", args={})
    cp, llm, tools = make_copilot({"plan": [same, same, same]})
    r = cp.ask("is it up?")
    assert len(tools.calls) == 1
    assert any(e["type"] == "repeated_action" for e in r["events"])


def test_invalid_planner_output_falls_back_to_retrieval(make_copilot):
    cp, llm, tools = make_copilot({"plan": ["this is not json"]})
    r = cp.ask("What does double-entry mean?")
    assert any(e["type"] == "invalid_plan" for e in r["events"])
    assert r["timings"][1]["node"] == "retrieve"


def test_unknown_tool_is_never_called(make_copilot):
    cp, llm, tools = make_copilot({"plan": [plan(action="tool", tool="create_transfer", args={"amount": 500})]})
    r = cp.ask("What does double-entry mean?")
    assert tools.calls == []
    assert any(e["type"] == "unknown_tool" for e in r["events"])


def test_invented_citation_numbers_are_dropped_and_uncited_answers_refused(make_copilot):
    cp, llm, tools = make_copilot({"answer": ["Postgres restarts itself [7]."]})  # only 3 chunks exist
    r = cp.ask("What are the recovery steps after a postgres crash?")
    assert r["refused"] and r["reason"] == "no_valid_citation"
    assert any(e["type"] == "invalid_citation" and e["numbers"] == [7] for e in r["events"])


def test_verifier_no_turns_the_answer_into_a_refusal(make_copilot):
    cp, llm, tools = make_copilot({"verify": ["no"]})
    r = cp.ask("What are the recovery steps after a postgres crash?")
    assert r["refused"] and r["reason"] == "verifier_unsupported"
    assert r["answer"] == REFUSAL


def test_model_saying_it_does_not_know_is_a_refusal(make_copilot):
    cp, llm, tools = make_copilot({"answer": ["I don't know from the docs."]})
    r = cp.ask("What is the capital of France?")
    assert r["refused"] and r["reason"] == "model_said_no_support"


def test_answer_prompt_contains_numbered_chunks_and_ledger_data(make_copilot):
    cp, llm, tools = make_copilot({"plan": [plan(action="retrieve"), plan(action="tool", tool="get_account",
                                                                          args={"account_id": 2}),
                                            plan(action="answer")]})
    cp.ask("recovery steps and balance of account 2")
    answer_prompt = next(p for p in llm.prompts if p.startswith("TASK: answer"))
    assert "\n[1] (source: runbook.md § Recovery steps)" in answer_prompt
    assert '\n[T1] get_account({"account_id": 2}): ' in answer_prompt
    assert "₹375.00" in answer_prompt  # not escaped to \\u20b9
