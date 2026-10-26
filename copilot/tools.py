# The agent's tools: five READ-ONLY calls to the real ledger API (project2-payment-ledger).
# Read-only by construction: the only HTTP helper here is _get(), so no code path in the copilot
# can POST a transfer, even if the model asks for one. The guardrails block write requests too:
# two independent layers (defence in depth).
# Each tool returns a plain dict and never raises, so a ledger outage becomes an observation the
# model can report ("the ledger returned 503") instead of crashing the request.

import os

import httpx

LEDGER_URL = os.getenv("LEDGER_URL", "http://localhost:8001")
# Our own client id, so the ledger's per-client rate limit counts the copilot separately.
_http = httpx.Client(base_url=LEDGER_URL, timeout=5, headers={"X-Client-Id": "ledgerops-copilot"})


def _get(path: str, params: dict | None = None) -> dict:
    try:
        r = _http.get(path, params=params)
    except httpx.HTTPError as e:  # connection refused, timeout...
        return {"ok": False, "error": f"ledger unreachable: {type(e).__name__}"}
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "error": body}
    return {"ok": True, "data": body}


def rupees(paise: int) -> str:
    """Convert in code, not in the model: a 7B model can get "37500 paise" -> rupees wrong."""
    sign = "-" if paise < 0 else ""
    return f"{sign}₹{abs(paise) // 100:,}.{abs(paise) % 100:02d}"


def _add_rupees(obj):
    """Next to every *_paise field, add the same amount as a ₹ string."""
    if isinstance(obj, list):
        return [_add_rupees(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _add_rupees(v)
            if k.endswith("_paise") and isinstance(v, int):
                out[k.removesuffix("_paise") + "_rupees"] = rupees(v)
        return out
    return obj


def _positive_int(value) -> int:
    n = int(value)  # raises ValueError/TypeError for "abc" or None
    if n < 1:
        raise ValueError("must be >= 1")
    return n


def get_health() -> dict:
    return _get("/health")


def get_stats() -> dict:
    return _add_rupees(_get("/stats"))


def get_account(account_id) -> dict:
    return _add_rupees(_get(f"/accounts/{_positive_int(account_id)}"))


def get_transfer(transfer_id) -> dict:
    return _add_rupees(_get(f"/transfers/{_positive_int(transfer_id)}"))


def get_statement(account_id, limit=5) -> dict:
    # A small default limit keeps the tool output (and so the prompt, latency and tokens) small.
    limit = min(_positive_int(limit), 20)
    return _add_rupees(_get(f"/accounts/{_positive_int(account_id)}/statement", {"limit": limit}))


# The allowlist the agent may call. Anything not in here does not exist as far as the model is concerned.
TOOLS = {f.__name__: f for f in (get_health, get_stats, get_account, get_transfer, get_statement)}

TOOL_DESCRIPTIONS = """\
- get_health(): is the ledger API up and its database reachable?
- get_stats(): ledger-wide audit: the 3 invariants (total balance, mismatched accounts, unbalanced transfers), \
healthy flag, and counts of customer accounts, transfers, ledger entries, idempotency keys
- get_account(account_id): one account's owner name and balance
- get_transfer(transfer_id): one transfer: from, to, amount, and its DEBIT and CREDIT entries
- get_statement(account_id, limit): an account's latest ledger entries (newest first) with running balance"""


def call_tool(name: str, args: dict) -> dict:
    """Run one allowlisted tool. Bad names or arguments become an error result, never an exception."""
    if name not in TOOLS:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    try:
        return TOOLS[name](**args)
    except (TypeError, ValueError) as e:  # wrong/missing argument names or values
        return {"ok": False, "error": f"bad arguments for {name}: {e}"}
