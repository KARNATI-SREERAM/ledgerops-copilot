# Guardrails: cheap, deterministic checks that run OUTSIDE the model, so a clever prompt can't
# talk its way past them.
#   detect_injection(text) : "ignore previous instructions"-style text, in the user's question AND
#                            in every retrieved chunk (a poisoned doc is an attack too)
#   detect_write(text)     : requests to move money or change data ("transfer 500 to account 3")
#   MIN_SCORE              : below this cosine score, the best chunk is not about the question
# Regexes are simple and explainable, but a paraphrase can slip past them. That is why the tools
# are also read-only by construction (tools.py): a missed write request still can't move money.

import re

# Calibrated with scripts/calibrate_threshold.py on questions that are NOT in the eval set
# (results/calibration.txt): on-topic min 0.678, off-topic max 0.654, so 0.66 sits between them.
MIN_SCORE = 0.66

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b[^.?!\n]{0,30}\b(instructions?|rules?|prompts?|guardrails?|"
    r"guidelines|polic(y|ies)|restrictions?)\b",
    r"\byou are now\b",
    r"\b(pretend|act) (to be|you are|as)\b",
    r"\b(developer|god|admin|debug|jailbreak|dan) mode\b",
    r"\bjailbreak",
    r"\b(system|hidden|initial) prompt\b",
    r"\b(reveal|show|print|repeat|leak)\b[^.?!\n]{0,20}\b(your|the) (instructions|rules|prompt)\b",
    r"\bnew instructions?\s*:",
    r"^\s*(system|assistant)\s*:",  # a line pretending to be a system or assistant message
]

AMOUNT = r"(₹\s?[\d,]+|\brs\.?\s?[\d,]+|\binr\s?[\d,]+|\b[\d,]+\s?(rupees?|paise|rs)\b|\b\d[\d,]*\b)"
WRITE_PATTERNS = [
    # money movement: a verb, an amount, then a destination or source ("transfer 500 to account 3")
    r"\b(transfer|send|pay|move|wire|withdraw|deposit|credit|debit|add|deduct)\b[^.?!\n]{0,40}?" + AMOUNT
    + r"[^.?!\n]{0,40}?\b(to|into|from)\b",
    # undoing money movement ("refund transfer 9")
    r"\b(refund|reverse|cancel|undo|roll ?back|void|chargeback)\b[^.?!\n]{0,30}"
    r"\b(transfers?|payments?|transactions?)\b",
    # changing ledger data ("set the balance of account 4 to ...", "delete transfer 12", "create an account")
    r"\b(create|open|close|delete|remove|freeze|unfreeze|update|modify|change|edit|set|reset|increase|decrease)\b"
    r"[^.?!\n]{0,40}\b(accounts?|balances?|transfers?|ledger|entries|entry|records?|idempotency)\b",
    # raw SQL writes
    r"\b(insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|truncate\s+\w+|alter\s+table)\b",
]

# "How do I open an account?" or "What happens if I send 500 to account 3 twice?" ask for information,
# not for an action, and the copilot can only read anyway. "Can you/please ..." are requests and stay blocked.
INFORMATIONAL = re.compile(r"^\s*(how|why|what|when|which|where|explain|describe|does|do|is|are|was|were)\b", re.I)

_injection = [re.compile(p, re.I | re.M) for p in INJECTION_PATTERNS]
_write = [re.compile(p, re.I) for p in WRITE_PATTERNS]

WRITE_REFUSAL = ("I can't do that. LedgerOps Copilot is read-only: it can look up health, stats, accounts, transfers "
                 "and statements, but it never moves money or changes data. Use an authorised client of the ledger "
                 "API (POST /transfers with an Idempotency-Key) for that.")
INJECTION_REFUSAL = ("I can't follow that request: it looks like a prompt-injection attempt (it tries to override my "
                     "instructions). Ask me a question about the ledger or its docs instead.")


def detect_injection(text: str) -> str | None:
    """Return the matched pattern if `text` tries to override instructions, else None."""
    for p in _injection:
        if p.search(text):
            return p.pattern
    return None


def detect_write(question: str) -> str | None:
    """Return the matched pattern if the question asks to move money or change data, else None."""
    if INFORMATIONAL.match(question):
        return None
    for p in _write:
        if p.search(question):
            return p.pattern
    return None
