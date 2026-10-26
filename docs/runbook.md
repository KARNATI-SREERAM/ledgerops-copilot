# Runbook: payment ledger (project2-payment-ledger) operations

This runbook is written from the ledger's real failure drill (`benchmark/failure_drill.py`, recorded in
`results/measured_run.txt` section 5) and from one real startup incident seen on 2026-10-26.
Every number below comes from those recordings; nothing is estimated.

## Service overview

- Two containers from `docker compose up --build` in `project2-payment-ledger/`: `postgres` (PostgreSQL 16, host port 5434)
  and `api` (FastAPI, host port 8001).
- Source of truth: the `ledger_entries` table. `accounts.balance_paise` is a cache that the audit can verify.
- All money is integer paise. The system account `external_funding` (id 1) holds minus all customer money,
  so the sum of every balance is always 0.
- Read-only endpoints: `GET /health`, `GET /stats`, `GET /accounts/{id}`, `GET /accounts/{id}/statement`, `GET /transfers/{id}`.
- Write endpoints: `POST /accounts` and `POST /transfers` (requires an `Idempotency-Key` header).

## Health checks

- `curl localhost:8001/health` returns `{"status": "ok"}` only if the API can run `SELECT 1` on the database.
  It is not rate limited, so monitors can call it often.
- `docker compose ps` shows both containers. Both define a healthcheck: `pg_isready` for postgres and a Python call to `/health` for api.
- `GET /stats` returns the three ledger invariants plus counts (customer accounts, transfers, ledger entries, idempotency keys).
  A healthy ledger has `total_balance_paise = 0`, `accounts_mismatched = 0` and `transfers_unbalanced = 0`.

## Incident: database crash during transfers (failure drill)

### What the drill does
A client sends 300 transfers of ₹1 from account A to account B, one at a time, each with its own Idempotency-Key.
After the 100th transfer, a thread SIGKILLs the Postgres container (no clean shutdown, like pulling the power cable)
and starts it again 2 seconds later.

### Symptoms
- `POST /transfers` returns **HTTP 503** with `{"detail": "database unavailable"}`. The API maps every SQLAlchemy
  `OperationalError` (DB down, connection lost, pool timeout) to 503, meaning "retry later", not 500 "bug".
- In the recorded drill: 12 failed attempts, all of them 503 (`by_type={503: 12}`).
- `GET /health` also fails while the database is down.

### Measured impact (recorded run)
- Postgres SIGKILLed at t=0.34 s and restarted at t=2.68 s.
- First failed request at t=0.41 s, first success after it at t=3.01 s: a **client-visible outage of 2.60 s**.
- 313 attempts for 300 transfers. Total drill time was 4.71 s.
- Result: exactly 300 transfers recorded from A. Balance of A was 970000 paise and B was 30000 paise, as expected.
  Nothing was lost and nothing was doubled. The ledger audit afterwards showed all invariants at 0.

### Recovery steps
1. **Confirm the outage.** `curl localhost:8001/health` fails or `docker compose ps` shows postgres not running or unhealthy.
2. **Restart the database if it did not come back by itself.** `docker compose start postgres` (or `docker compose up -d`).
   Wait for `pg_isready` to report healthy.
3. **No API restart is needed.** The connection pool uses `pool_pre_ping=True`: each pooled connection is tested before use,
   so dead connections from before the crash are replaced automatically and the API reconnects by itself.
4. **Clients retry with the SAME Idempotency-Key.** On a 503, a connection error or a timeout, the client must resend the same
   request with the same key until it gets a final answer (201 created or 422 refused). Using a new key for a retry could pay twice.
   The key is claimed in the same database transaction as the money movement, so a transfer is either fully committed with its key or not at all.
5. **Back off between retries.** The drill client sleeps 0.2 s between retries. On HTTP 429 (rate limited) wait for the `Retry-After` header instead.
6. **Verify the ledger.** Run `.venv/bin/python scripts/check_ledger.py` (or call `GET /stats`). It prints the invariants and counts and
   exits with status 1 if any invariant is broken. Expected after recovery: `total_balance_paise=0 accounts_mismatched=0 transfers_unbalanced=0`.
7. **Reconcile the client side.** For the affected accounts, compare `GET /accounts/{id}/statement` with what the client believes it sent:
   the number of DEBIT entries should equal the number of transfers the client got a 201 for.

### Why no money is lost or doubled
- At-least-once delivery (the client retries until it gets an answer) plus idempotent processing (a replayed key returns the stored
  response and moves nothing) gives an exactly-once **effect**.
- PostgreSQL's write-ahead log means every committed transaction survives a SIGKILL. Uncommitted ones are rolled back.
- Double-entry bookkeeping (one DEBIT plus one CREDIT per transfer) lets the audit prove the books still balance.

## Incident: API crash-loops at startup ("Name or service not known")

### Symptoms
- `docker ps` shows the api container as `Restarting`.
- `docker logs <api container>` ends with
  `sqlalchemy.exc.OperationalError: (psycopg.OperationalError) [Errno -2] Name or service not known` and `Application startup failed. Exiting.`
- `curl localhost:8001/health` gives "connection refused" (nothing is listening).

### Cause
At startup the API runs `init_db()`, which needs the database. The hostname `postgres` could not be resolved: the postgres
container had exited (seen on 2026-10-26: it had been stopped for 8 days), so the API exits and Docker restarts it again and again
(`restart: unless-stopped`).

### Recovery steps
1. `docker compose up -d` in `project2-payment-ledger/` to start postgres again. Wait until it is healthy.
2. If the api container keeps restarting with the same error after postgres is healthy, recreate it so it rejoins the network:
   `docker compose up -d --force-recreate api`. This was needed in the 2026-10-26 incident.
3. Check `curl localhost:8001/health` returns `{"status": "ok"}`, then run the ledger audit (`GET /stats` or `scripts/check_ledger.py`).

## Incident: clients get HTTP 429 "rate limit exceeded"

- Each client (the `X-Client-Id` header, else the IP address) has a token bucket: a burst of 100 requests, refilling at 50 requests/s by default
  (`RATE_LIMIT_BURST`, `RATE_LIMIT_PER_SEC`).
- The response carries a `Retry-After` header in seconds. Clients must wait that long. A retried transfer must reuse its Idempotency-Key.
- `/health`, `/docs` and `/openapi.json` are never rate limited.
- In the recorded run, one client sending 300 requests back to back got 178 through in 1.58 s, within the bound burst + rate × time = 179.
  A different client was not affected.

## Incident: a transfer is refused with HTTP 422

- `insufficient funds`: the sender's balance is lower than the amount. The refusal is stored against the Idempotency-Key, so retrying the
  same key stays refused even if money arrives later. Send a new payment with a new key instead.
- A key reused with a different request body also returns 422: that is a client bug (one key must mean one payment).

## Escalation checklist

- Invariants not 0 after recovery: stop accepting writes and escalate. A non-zero `accounts_mismatched` means a cached balance disagrees with its entries.
- Repeated 503s while postgres is healthy: check the pool (10 + 5 connections, 5 s `pool_timeout`) and database locks.
