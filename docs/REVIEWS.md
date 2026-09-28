# Final reviews

Three reviews — security, performance, architecture — with what each found and
what was done about it. Findings are recorded with their disposition, including
the ones deliberately **accepted**, because a review that only reports what it
fixed is a review that cannot be audited.

Every "verified" below names the check that verified it. "Looks fine" is not a
finding, and the difference between the two is whether someone can re-run it.

---

## Security review

### Fixed

**0. A configured limit that bounded nothing.** `MAX_TOKEN_BUDGET` was read from
settings and never used, and the `prompt_tokens` / `completion_tokens` columns on
tasks and agent runs were always zero: usage was parsed off every model response
and then dropped. The README's own claim — that a bound you cannot name is not a
bound — was false in exactly the way it warns about.

Three pieces now: a `TokenBudget` accumulator, a `BudgetedProvider` decorator that
charges every call including the router's and the critic's, and a
`ContextVar`-scoped budget per run so two concurrent runs cannot pool their counts.
Enforcement happens *before* a call, so a completion already paid for is not
discarded; the run aborts loudly rather than continuing without model access and
inventing an answer.

Writing it surfaced two more defects:

- **`/api/v1/chat` had no budget at all.** Only the asynchronous task path scoped
  one, so the setting was enforceable through one of two doors. Both now build the
  budget through one shared helper, so they cannot diverge again.
- **The chat handler flattened deliberate refusals into 500s.** Its catch-all
  converted every `AppError` — a spent budget, a provider rejection — into a
  server error, telling the caller to retry something that would fail
  identically. `AppError` now passes through to the registered handler and keeps
  its own status.

*Checked by* `tests/unit/test_budget.py`, `tests/graph/test_graph_execution.py`,
and `tests/api/test_api.py`, which asserts the persisted token counts, the 429 on
the chat path, and a failed task with a recorded reason.

**1. The OpenAPI schema was served in every environment.**

`/docs` and `/openapi.json` were enabled unconditionally. The schema is an
inventory of every route, every field, and every error shape: useful while
building against the API, and a map of the attack surface once deployed.

`Settings.serve_api_docs` now decides it — `True` outside production, `False`
inside it, with `API_DOCS_ENABLED` overriding either way so a deployment behind
an authenticating gateway can still expose them deliberately. Both the schema and
the page that reads it are removed together; leaving `/docs` without a schema
would serve a broken page rather than none.

*Checked by* `tests/unit/test_config.py::test_api_docs_are_withheld_in_production_by_default`
and `tests/api/test_dashboard.py::test_the_schema_can_be_withheld`.

**2. The operator console needed to be off by default, and safe when on.**

A page with approve, reject, and cancel buttons is a control surface. It is now
gated on `DASHBOARD_ENABLED` (default `False`) and designed so that its own
weakest link is not user input: no inline script or style, so the CSP runs with
`script-src 'self'` and no `unsafe-inline`; and every value from the API is
written with `textContent`, never `innerHTML`.

*Checked by* `tests/api/test_dashboard.py`, including a structural guard that the
script contains no HTML-building API — and that the guard itself can fail, so it
cannot decay into a tautology.

### Verified

**No string-built SQL.** No `execute(f"…")` or `text(f"…")` anywhere in `app/`.
The only dynamic statements are the fixed-identifier `TRUNCATE` in the test
fixtures and PostgreSQL's own `pg_terminate_backend` housekeeping.

**No shell.** No `subprocess`, `os.system`, or `shell=True` in `app/`. The only
`eval` is Redis's `EVAL` for the atomic compare-and-delete lock release, which is
a Lua script passed as a literal.

**No secret in a log.** The authentication path logs `type(exc).__name__` and
nothing else, because a decoder's message can echo the token back. The only
values logged by the cache layer are keys, and keys are namespaced identifiers,
not credentials.

**Execution stays refused.** `python_executor` raises unless a sandbox is
configured, `execution_sandbox_available` returns `False` in every environment by
construction, and the production validator refuses to start when execution is
enabled. There is no configuration that turns it on safely on this host, which is
the honest state of affairs rather than a gap to paper over.

**Cancellation is never swallowed.** Seventeen `except Exception` sites were
reviewed. All are at transport or process boundaries where the alternative is a
crash, and all are safe with respect to cancellation because
`asyncio.CancelledError` derives from `BaseException` and therefore passes
through.

**Cross-origin reads are denied by default.** No CORS middleware is installed, so
a browser will not hand a response to another origin. Nothing in the system needs
one.

**Rate limiting cannot be evaded by being unattributable.** The bucket is chosen
from the token subject, then the trusted identity header, then the client
address, and finally a single shared anonymous bucket — an unattributable request
is limited rather than exempt. Unmatched paths are counted under a constant route
label, so a caller cannot mint unbounded metric series by requesting nonsense.

### Accepted risks

**A provider's `Retry-After` in HTTP-date form is ignored.** The header permits
both a delta and an absolute date, and only the delta is honoured. Converting a
date would mean trusting the provider's clock against this machine's, where a few
seconds of skew silently turns a short wait into a long one. Ignoring it falls
back to the backoff curve, which at least fails in a known direction.

**Child-row reads have no `LIMIT`.** `TaskStepRepository.list_for_task`,
`AgentRunRepository.list_for_task`, and `ToolCallRepository.list_for_task` are
bounded by `task_id` only. The graph caps iterations, tool calls, and wall-clock
time per task, so the row count is bounded by configuration rather than by the
query. Adding a limit would be a second, weaker copy of a bound that already
exists, and the weaker copy is the one that would go stale.

*Revisit if* the iteration ceiling is ever removed or made permissive.

**The console's HTML is served unauthenticated.** Only its API calls require a
token, so the page renders an empty shell without one — but the shell itself is
still served. This is stated in the route module, in `.env.example`, and in the
README rather than left for an operator to discover, and the console is off
unless it is switched on.

---

## Performance review

### Fixed

**A throttle instruction was thrown away.** A provider that answers 429 or 503
with `Retry-After` is saying how long to wait, and the value was discarded: the
retry loop waited its own backoff, which for a short curve means retrying
immediately and earning another rejection. The instruction is now parsed from the
header and honoured as a *floor* beneath the curve, capped at a minute.

*Checked by* `tests/unit/test_retry_after.py`. Writing it found an edge case in
the first implementation: `NaN` passes a bare sign check, so a nonsense header
would have reached `asyncio.sleep` and turned a throttle into a crash. The check
is now an explicit finiteness test.

**Memory scoring recomputed three invariant values per candidate.** Retrieval
scans up to `memory_scan_limit` rows, and each row re-tokenised the query text,
re-took the norm of the query embedding, and re-read the clock. All three are
properties of the query, not of the candidate.

Hoisting them into a prepared context makes a 500-candidate scan **4.8× faster**
(10.3 ms → 2.2 ms on the development host). The harness asserts equivalence as
well as speed: scores agree to within `1e-6`, and the best-scoring memory is
unchanged. The tolerance is not hand-waving — reading the clock once instead of
once per row shifts each recency term by about a nanosecond, and exact equality
would have reported a false alarm on a correct change.

*Checked by* `tests/evaluation/test_benchmarks.py`, which asserts the ratio, the
score agreement, the ranking, and a budget an order of magnitude above the
measured cost.

### Verified

**Every repository list is bounded.** Conversations 50, tasks 50 (clamped to 200
at the route), tasks-by-status 100, pending approvals 100, events 500, memories
200, prune batch 1000.

**Indexes match query shapes.** Composite indexes exist for the (user, status,
created-at), (task, created-at), and (user, kind, created-at) access patterns the
repositories actually issue, added in the same phase as the queries that need
them.

**Measured hot paths.** Tokenise 0.012 ms, estimate importance 0.025 ms, prepare a
query context 0.018 ms, score 500 vector candidates 2.15 ms, score 500 lexical
candidates 3.37 ms.

---

## Architecture review

### Found, and fixed

**The retry policy was written, tested, and never called.** A classification
table, a backoff curve, a retry loop with injectable sleep, a suite of 24 tests
covering all three — and no caller anywhere in `app/`. A transient 429 or a
timeout failed the whole run while the machinery to survive it sat unused beside
the code path. Every individual test passed throughout, which is exactly why the
gap survived: each part was correct, and nothing asserted the parts were joined.

The join is now `RetryingProvider`, a decorator applied outside the budget
decorator, with the ordering rationale written into the module so the next person
does not reverse it by accident. The new tests assert the *composition*, not the
pieces: a transient failure earns another attempt, a permanent one does not, a
spent budget is not retried, and the original failure is what escapes rather than
the loop's own wrapper type.

**A decorator changed what a caller is told.** Retrying inside the provider means
the failure a caller classifies is the provider's own error, not
`RetryExhaustedError` — whose kind is not one the classification table knows.
The original is re-raised with the exhausted wrapper attached as the chain's
cause, so the log still explains how many attempts were spent.

### Verified

**Dependency direction holds.** No module under `app/services`, `app/graph`,
`app/tools`, or `app/agents` imports `app.api`. The only `app/api` import of
`app.database` is the dependency factory that constructs the connection — that is
the injection seam, not a route reaching into storage.

**No per-run state on shared objects.** The compiled graph is built once and
invoked concurrently, so anything stored on it would interleave two runs' events
into one task's history. Per-run routing goes through a `ContextVar`
(`event_sink_for`) instead. This was a real defect, found by reasoning about
concurrency rather than by a test, and it is now structural.

**The documentation matches the system.** The phase tables in
`docs/DEVELOPMENT_PLAN.md` and the roadmap in the README were stale — they listed
completed phases as blocked and not-started. Both are corrected, and the
correction is asserted indirectly: the README's test count and file counts are
taken from a real run, and the diagram audit re-derives the graph's node names
from the code.

**The diagrams cannot drift.** They are generated by `scripts/generate_assets.py`,
which audits every drawn label for overlap and overflow and fails the build on a
defect, and CI fails when the committed PNGs differ from what the script
produces. A diagram that claims a node the graph does not have is a test failure,
not a documentation bug nobody notices.

### Found by the suite, and recorded

**Two systems own parts of one schema.** The new migration-drift check failed the
first time the whole suite ran together and passed when it ran alone. The cause
was real: the checkpointer library creates and owns four tables of its own
(`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`)
at start-up, and they only exist once something has opened a checkpointer.

The arrangement is deliberate and stays. The checkpointer must be free to change
its own schema with its own version; duplicating those tables into an Alembic
revision would put two migration mechanisms in charge of one schema and guarantee
they disagree. What changed is that the drift check now states the split instead
of tripping over it: it requires nothing outside `alembic_version` and the
checkpointer's four tables, and treats the four as all-or-nothing, so an
interrupted `setup()` is still a failure.

*Checked by* `tests/integration/test_migration_drift.py`, which is a test that
found a real gap on its first full run — the reason it was worth writing.

### Accepted

**`app/evaluation` ships inside the application package.** It is a development
tool: a corpus, a metrics module, a harness, and benchmarks. Placing it under
`app/` means it is importable in the image, which is a small cost, and it means
the harness can be pointed at the *real* router and the *real* scoring function
instead of a copy of them. A copy would measure the copy.

---

## Verification

| Phase | Scope | Status |
| :--- | :--- | :--- |
| 43 | Security review | Complete — one finding fixed, one accepted and documented |
| 44 | Performance review | Complete — one optimisation shipped, budgets asserted |
| 45 | Architecture review | Complete — no layer violations; stale documentation corrected |

Migration verification, which was blocked alongside the container work, was
completed without a container runtime: the revision applies to an empty database,
reverses cleanly, and re-applies; Alembic's own autogenerate comparison reports no
drift; and `tests/integration/test_migration_drift.py` compares the catalog the
migrations built against the metadata the models declare — tables, columns,
primary keys, and indexes.
