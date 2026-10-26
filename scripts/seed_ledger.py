# Creates known eval data in the real ledger THROUGH ITS API, and writes the ids to evals/seed.json.
# This is setup for the eval, not part of the copilot (the copilot can only read). Fresh accounts
# every run, so the expected balances below are always exact, whatever else is in the database.
# Run: .venv/bin/python scripts/seed_ledger.py      (the ledger must be up on LEDGER_URL)

import json
import os
import uuid
from pathlib import Path

import httpx

LEDGER_URL = os.getenv("LEDGER_URL", "http://localhost:8001")
SEED_FILE = Path(__file__).parent.parent / "evals" / "seed.json"

# Final balances: A = 5,000.00 - 1,250.00 = ₹3,750.00; B = 1,250.00 + 375.50 = ₹1,625.50;
# C = 2,500.00 - 375.50 = ₹2,124.50
ACCOUNTS = {"A": ("Asha-eval", 500_000), "B": ("Ravi-eval", 0), "C": ("Meera-eval", 250_000)}
TRANSFERS = {"T1": ("A", "B", 125_000), "T2": ("C", "B", 37_550)}


def seed() -> dict:
    http = httpx.Client(base_url=LEDGER_URL, timeout=10, headers={"X-Client-Id": "eval-seed"})
    ids = {}
    for key, (name, paise) in ACCOUNTS.items():
        r = http.post("/accounts", json={"owner_name": name, "initial_balance_paise": paise})
        r.raise_for_status()
        ids[key] = r.json()["id"]
    for key, (src, dst, paise) in TRANSFERS.items():
        r = http.post("/transfers", headers={"Idempotency-Key": str(uuid.uuid4())},
                      json={"from_account_id": ids[src], "to_account_id": ids[dst], "amount_paise": paise})
        r.raise_for_status()
        ids[key] = r.json()["id"]
    SEED_FILE.write_text(json.dumps(ids, indent=2) + "\n")
    return ids


if __name__ == "__main__":
    print(seed())
