# A minimal Streamlit chat page. It only calls the copilot's HTTP API (POST /ask) and shows the
# answer, its citations, the tools used and any refusal reason. No logic lives here, so the UI
# can't bypass a guardrail.
# Run locally: .venv/bin/streamlit run ui.py      (in compose: the `ui` service, http://localhost:8501)

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8002")

st.set_page_config(page_title="LedgerOps Copilot", page_icon="📒")
st.title("LedgerOps Copilot")
st.caption("Read-only assistant for the payment ledger: answers from the docs (with citations) "
           "and from live ledger data. It refuses when it can't ground an answer.")

if "history" not in st.session_state:
    st.session_state.history = []  # survives Streamlit's rerun of this script on every interaction


def show(result: dict) -> None:
    st.markdown(result["answer"])
    if result["refused"]:
        st.warning(f"Refused: {result['reason']}")
    if result["citations"]:
        with st.expander(f"Sources ({len(result['citations'])})"):
            for c in result["citations"]:
                st.markdown(f"- {c}")
    if result["tools_used"]:
        st.caption("Ledger tools used: " + ", ".join(result["tools_used"]))


for role, content in st.session_state.history:
    with st.chat_message(role):
        show(content) if role == "assistant" else st.markdown(content)

if question := st.chat_input("e.g. What is the balance of account 2?  /  What should a client do after a 503?"):
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking (local 7B model, ~5-20 s)..."):
            try:
                r = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=300)
                r.raise_for_status()
                result = r.json()
            except httpx.HTTPError as e:
                result = {"answer": f"The copilot API is not reachable ({type(e).__name__}).", "refused": True,
                          "reason": "api_error", "citations": [], "tools_used": []}
        show(result)
    st.session_state.history.append(("assistant", result))
