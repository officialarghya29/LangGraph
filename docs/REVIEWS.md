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

### Closed

**Child-row reads had no `LIMIT`.** `TaskStepRepository.list_for_task`,
`AgentRunRepository.list_for_task`, and `ToolCallRepository.list_for_task` were
bounded by `task_id` alone. The argument for accepting that was that the graph
caps iterations, tool calls, and wall-clock time per task, so the row count is
bounded by configuration. The counter-argument won: that bound lives somewhere
else, and nothing would force the query to change if the ceiling were raised. All
three now take a ceiling and an offset, tested against real PostgreSQL.

**A provider's `Retry-After` in HTTP-date form was ignored.** It is now
interpreted against the response's *own* ``Date`` header rather than this
machine's clock — the header expresses a relative wait, and subtracting two
different clocks reintroduces exactly the skew the relative form exists to
avoid. Without a ``Date`` header, or when the instant has already passed, it is
still ignored in favour of the backoff curve.

**The adapters had never spoken HTTP to anything.** Every test of them replaced
the transport, which tests the *parsing* and cannot test the *request*: a wrong
base path, a header that never gets sent, a body a server would reject. A
scripted loopback server now answers them over a real socket, and the assertions
are about what arrived — the endpoint, the bearer token, the JSON body, the
Anthropic key header and its mandatory ceiling.

What this does **not** cover is vendor-specific behaviour: rate-limit quirks,
model-specific payloads, streaming, or anything else only the real endpoint does.
That still needs a credential and a network, and it is stated rather than
implied.

### Accepted risks

**The console's HTML is served unauthenticated.** Only its API calls require a
token, so the page renders an empty shell without one — but the shell itself is
still served. It stays that way deliberately: the alternative is a token in the
URL, which leaks into browser history, referrers, and access logs, and is worse
than serving static markup that contains no data. The deployment pattern is to
front it with an authenticating proxy, which the docs state. The console is off
unless it is switched on, and it carries `noindex`, `no-referrer`, and a policy
that permits nothing to be loaded from anywhere else.

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
200, prune batch 1000, and the three per-task child reads 500 each — see
[Closed](#closed) for why the last three moved out of this section.

**Indexes match query shapes.** Composite indexes exist for the (user, status,
created-at), (task, created-at), and (user, kind, created-at) access patterns the
repositories actually issue, added in the same phase as the queries that need
them.

**Measured hot paths.** Tokenise 0.012 ms, estimate importance 0.025 ms, prepare a
query context 0.018 ms, score 500 vector candidates 2.15 ms, score 500 lexical
candidates 3.37 ms.

**Cost is now observable, not just bounded.** Once usage was actually counted it
was worth exposing: `llm_calls_total`, `llm_tokens_total` by direction, and
`llm_retries_total` by failure kind. Cardinality is bounded by construction —
two directions, ten kinds — so a misbehaving provider cannot inflate the
exposition. Failed runs are counted too, because a run that died halfway consumed
what it consumed, and a cost dashboard that only counted successes would
understate exactly the runs worth investigating.

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

## Deep scan

A pass with one question: **which claims in this repository are not true?** Not
"what could be better" — that is what the three reviews above are for. The target
was the specific failure this project warns about elsewhere: configuration,
documentation, and tests that describe behaviour no code performs. Each finding
below was reachable by running the system, and each was fixed rather than
annotated.

### Fixed

**A ceiling that bounded nothing, twice.** `MAX_TOOL_CALLS` and
`MAX_EXECUTION_TIME` were declared, given bounds, documented as hard ceilings in
four places, and read by nothing — while the `TOOL_STARTED`/`TOOL_COMPLETED`
event types, the `tool_call_count` column, and the `ExecutionLimitError` type all
existed unused, waiting for them. A run could call tools and consume wall-clock
time without limit.

`app/services/limits.py` now holds the enforcement: a `RunLimits` in a
`ContextVar` per run, checked at `BaseAgent.call_tool` — the single path every
invocation passes through — and in the dispatch loop's exit condition, with a
hard `asyncio.timeout` around the graph as the backstop for a run that is inside
one long call when the deadline passes. Refusals happen before the work, never
after it. Verified by `tests/unit/test_limits.py`, the ceiling cases in
`tests/agents/test_agents.py`, and an end-to-end case in
`tests/graph/test_graph_execution.py` where an expired run cannot perform its
approved action and says so.

**Three audit tables were empty for every run this system had ever executed.**
`record_step`, `finish_step`, and `record_tool_call` were declared on the store,
implemented, and tested directly — and called by no production code. The module
docstring promising that "steps, agent runs, tool calls, approvals, and events
land in their own tables" was two-fifths true. The run's sink now writes the
event log *and* the specialised rows, which is why the agent announces its own
tool calls: a worker calls tools from inside its own `run`, and the node above it
only ever sees the result.

**A count that was always zero.** `tool_call_count` and `iteration_count` were
updatable columns that nothing updated, so a task making twenty tool calls
reported none. Combined with the item above, this is the same disease in two
places: a number that looks authoritative and is fictitious. The counts now come
from the run — including a failed or timed-out run, which is precisely the run
whose consumption someone will want to see — and the approved path *adds* its
segment rather than replacing the earlier count.

**A blank credential registered a tool that could only fail.** `GITHUB_TOKEN=`
in a `.env` file parses as an empty secret, which is not `None`, so the registry's
"is a token configured?" test passed and listed two GitHub tools that would fail
authentication on first use — an error that reads like an upstream problem rather
than a missing setting. Worse, the same distinction let a deployment start
believing it had a `JWT_SECRET`: the production guard is an `is None` check, and
an empty string is not `None`. Empty and whitespace-only secrets and URLs are now
treated as absent. The API suite had pinned the resulting tool count as a literal
`7`; the assertion now compares the gauge against discovery instead.

**Two settings declared and never read.** `LLM_MAX_RETRIES` left the provider
retrying on the graph's *replan* budget — one number doing two unrelated jobs —
and `TRACE_HISTORY_SIZE` left the tracer at its own default. Both are wired, and
both are asserted by reading the running object rather than the setting.

**A kill switch that killed nothing.** `METRICS_ENABLED=false` changed what
`/ready` reported and left `/metrics` serving. It now returns 404.

**A console whose live stream could not work.** The dashboard's `EventSource`
cannot set request headers, so the identity and token were dropped and the stream
failed; passing them in the query string would have worked and would have written
credentials into browser history, referrers, and access logs. The stream is now
read through `fetch` with an `AbortController` — which can send headers — plus a
Stop control. Two tests keep it honest: every element the script reaches for must
exist, and the page must not use `EventSource` or put a credential in a URL.

**Two defects found by the fixes above, not by the scan.** `RiskLevel` is an
`IntEnum`, so publishing `.value` would have sent `1` where the `tool_calls` audit
column requires a level *name* — rejected by its own check constraint. And a
resumed run bound no event sink at all, so a client following a gated task saw it
fall silent at exactly the point where the action was performed.

### Found by the second pass, and fixed

**A blank value that crashed the process.** `API_DOCS_ENABLED=` is how an
operator says "decide from the environment". It is also not parseable as a
boolean, so the value was rejected and the application refused to start — over a
setting that was left deliberately blank, in a template that shipped it blank.
The tri-state flag now treats an empty string as unset.

**A blank value that configured nothing.** `LLM_MODEL=` is perfectly valid input
for a ``str`` field, so it was accepted, and every model call was then made with
an empty model name — an error about the request, not about the configuration,
from a file that looked correctly filled in. Blank names now fall back to their
declared defaults, and blank optional strings to "unset", which is the same rule
the secrets already followed.

**A settings template that documented 41 of 66 settings.** The missing ones
included the security-relevant switches — `TRUST_IDENTITY_HEADER`,
`RATE_LIMIT_FAIL_CLOSED`, `ALLOW_PRIVATE_NETWORK_EGRESS`, `JWT_AUDIENCE`,
`METRICS_ENABLED` — so the settings an operator most needs to know exist were
exactly the ones absent from the file they copy. It also advertised
`EMBEDDING_PROVIDER` as accepting ``cloud``, which the code has never accepted.

The template is now complete and annotated, and the suite holds it there: every
documented key must be a real setting, every real setting must be documented, and
the file must parse as working configuration. That check has a second purpose,
because the settings layer ignores unknown keys — a typo in a key name is accepted
in silence and the setting it meant to configure stays at its default while the
file says otherwise. The same check now covers `docker-compose.yml`, where the
same typo would be just as invisible and rather harder to notice.

**Two deployment artefacts that could not be checked by running them.** There is
no container runtime here, so the `Dockerfile` and `docker-compose.yml` remain
unbuilt — but the parts that *can* be checked statically now are: that every
variable compose sets exists, that its hostnames are the services it starts, that
it waits for them to be healthy, that the image copies the migrations its own
start-up command runs, that it runs as a non-root user, and that a real `.env`
cannot be baked into a layer. None of that is a substitute for building the image;
it is the subset that does not require one.

**A job that could hang for six hours.** CI had no timeout anywhere. A hung test
would hold a runner until GitHub's default cutoff and report the hang as a
timeout of its own. Every job is now bounded, and a committed-credential scan runs
over the full history, because a secret committed once lives in every clone and in
the history after the file is deleted.

### Verified, not a defect

**`except ValueError: pass` in the SSRF check.** The one place in the codebase
that swallows an exception, and it reads like a hole. It is the fast path that
tries to parse the host as a literal address before resolving it through DNS: a
`ValueError` here is the expected answer to "is this an IP, or a name?", and the
fallback is the next statement, not a silent failure. The DNS branch that follows
is what fails closed.

### Closed after the first pass

**Per-agent cost, and the `agent_runs` table with it.** This was recorded here as
an accepted gap: `record_agent_run` was called by nothing, so `agent_runs` was
empty, and the store's docstring promised a trail it did not write. The reason for
accepting it was real — the token budget is counted per run, and up to
`MAX_PARALLEL_TASKS` agents spend against it concurrently, so a delta taken around
one agent's call measures whatever happened to finish in that window. Wire that and
the columns named `prompt_tokens` hold numbers that are not prompt tokens, which is
the defect the whole section exists to remove.

It is closed rather than excused, because the correct instrument turned out to be
one that already existed in the codebase. The budget is scoped by a context
variable so that concurrent runs cannot pool their counts; scoping the *agent*
the same way answers the attribution question without a shared counter, because
`asyncio.gather` gives every coroutine its own copy of the context. Each
invocation therefore gets a fresh accumulator, every model call inside it is
charged once to the run and once to that invocation, and a retried subtask reports
its second attempt's cost rather than twice its first.

The scope is applied at the one place every agent invocation now passes through,
which also closed a second gap: the planner, critic, synthesizer, and executor were
running real model calls that the event stream never announced, so a client
following a gated task saw nothing between "approving" and "done".

Verified three ways: the unit suite asserts that two concurrent scopes cannot see
each other's usage and that a nested scope restores the outer one; the API suite
drives a full planned run — routing, a plan with two workers, criticism, synthesis
— and asserts one row per invocation with non-zero prompt and completion tokens for
every one of them; and a graph test runs two different shapes of run against a
single compiled graph at once, asserting that each stream received only its own
events and that a request costs the same whether or not it had company.

**A task timeline over HTTP.** The trail was readable only by opening a SQL
client. `GET /api/v1/tasks/{task_id}/timeline` now returns it, owner-scoped, and
the response schema is asserted in the suite — including the absence of tool
arguments, which are model-authored and can carry a credential.

---

## Verification

| Phase | Scope | Status |
| :--- | :--- | :--- |
| 43 | Security review | Complete — one finding fixed, one accepted and documented |
| 44 | Performance review | Complete — one optimisation shipped, budgets asserted |
| 45 | Architecture review | Complete — no layer violations; stale documentation corrected |
| — | Deep scan | Complete — thirteen findings fixed, one verified non-defect, five settings documented and two configuration defects closed |

Migration verification, which was blocked alongside the container work, was
completed without a container runtime: the revision applies to an empty database,
reverses cleanly, and re-applies; Alembic's own autogenerate comparison reports no
drift; and `tests/integration/test_migration_drift.py` compares the catalog the
migrations built against the metadata the models declare — tables, columns,
primary keys, and indexes.
