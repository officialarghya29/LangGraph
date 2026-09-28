# Technology Stack

## Environment as discovered (Phase 0)

Measured on the development machine on 2026-09-28.

| Item | Value |
| --- | --- |
| Operating system | Ubuntu 24.04 (kernel `7.0.0-34-generic`), x86_64 |
| Python | 3.12.3 (`/usr/bin/python3.12`) — the only interpreter present |
| pip | 26.2.1 (user site-packages) |
| `uv` | 0.12.5 (`~/.local/bin/uv`) |
| Git | 2.43.0, configured as `Arghya Bose <officialarghya29@gmail.com>` |
| Docker | **absent** |
| Docker Compose | **absent** |
| PostgreSQL (`psql`, `pg_config`, server) | **absent** |
| Redis (`redis-server`, `redis-cli`) | **absent** |
| PyPI reachability | reachable (HTTP 200) but slow and intermittently flaky |

### Consequences for the plan

- **Phases 3, 4, 38, 39 cannot be validated here.** PostgreSQL, Redis, and
  Docker are not installed, so database, cache, container, and migration work
  has no way to be verified on this machine. Those phases must either run
  against containers installed later, or against an external service.
- **Phase 31 (isolated Python execution) must default to disabled.** The
  requirement is that arbitrary code execution be sandboxed; without a container
  runtime and on a host we cannot isolate, the safe choice is
  `PYTHON_EXECUTION_ENABLED=false`. This matches the master prompt's own
  instruction to disable a capability rather than ship an unsafe implementation.
- **Python 3.12 is fixed.** It is the only interpreter available, so
  `requires-python` is `>=3.12,<3.13`.

## Selection principles

1. Every dependency must earn its place. No package is added because it is
   popular.
2. Prefer the standard library where it is sufficient.
3. Do not assume versions. Resolve against the interpreter actually present,
   then record what was installed.
4. Minimal stable dependency set; resolve conflicts by identifying the smallest
   compatible combination rather than downgrading packages at random.

## Core stack

| Concern | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.12 | Only interpreter available; fully supported by the stack |
| Environment / deps | `uv` | Available on the host, fast resolver, lockfile support |
| Packaging | `pyproject.toml` + Hatchling | Single source of truth for metadata and tool config |
| API framework | FastAPI | Async-native, Pydantic-validated, OpenAPI generated |
| ASGI server | Uvicorn | Standard companion server for FastAPI |
| Validation | Pydantic v2 | Runtime validation and settings; also the schema source for LLM structured output |
| Configuration | Pydantic Settings | Strongly typed config with `.env` support *(Phase 2 — done)* |
| Orchestration | LangGraph | The graph runtime the architecture is built around *(Phase 16 — done)* |
| LLM integration | In-house provider abstraction over direct HTTP. LangChain is used only for graph primitives, not for model access | Keeps agents vendor-independent, and avoids coupling the model layer to a framework's churn *(Phase 5 — done)* |
| Database | PostgreSQL + SQLAlchemy + Alembic | Durable tasks, approvals, memory, and checkpoints *(Phases 3, 21)* |
| Cache / coordination | Redis | Rate limiting and short-lived coordination; never sole durable storage *(Phase 4)* |
| Testing | pytest (+ `pytest-asyncio`) | Standard; async support needed for graph and API tests |
| Linting / formatting | Ruff | One tool for lint and format |
| Type checking | MyPy (strict) | Typed codebase is an explicit requirement |

## Resolved dependency versions

Resolved by `uv` on 2026-09-28 against CPython 3.12.3, and installed into
`.venv`. Direct dependencies are pinned exactly in `pyproject.toml`; this table
records the full transitive set that was actually tested.

### Direct

| Package | Version | Reason |
| --- | --- | --- |
| `fastapi` | 0.141.1 | HTTP framework with Pydantic validation and generated OpenAPI |
| `httpx` | 0.28.1 | HTTP client for the OpenAI and Anthropic provider adapters |
| `langgraph` | 1.2.12 | The graph runtime: typed state, checkpoints, interrupts |
| `pydantic` | 2.13.5 | Runtime validation; also the schema source for LLM structured output |
| `pydantic-settings` | 2.15.0 | Typed configuration with `.env` loading |
| `uvicorn[standard]` | 0.54.0 | ASGI server; extras add `httptools`, `uvloop`, `watchfiles`, `websockets`, `python-dotenv`, `PyYAML` |

### Direct (assets)

| Package | Version | Reason |
| --- | --- | --- |
| `pillow` | 12.3.0 | Renders the README diagrams. Build-time only |

### Direct (development only)

| Package | Version | Reason |
| --- | --- | --- |
| `httpx2` | 2.13.1 | Test client transport. Replaces `httpx`, which Starlette 1.7 deprecates |
| `mypy` | 2.3.1 | Strict static type checking |
| `pytest` | 9.1.1 | Test runner |
| `pytest-asyncio` | 1.4.0 | Async test support (`asyncio_mode = "auto"`) |
| `ruff` | 0.16.9 | Linting and formatting |

### Transitive

`annotated-doc` 0.0.5, `annotated-types` 0.8.0, `anyio` 4.15.1,
`ast-serialize` 0.11.2, `certifi` 2026.7.22, `click` 8.5.0, `h11` 0.16.0,
`httpcore2` 2.13.1, `httptools` 0.8.0, `idna` 3.20, `iniconfig` 2.3.0,
`librt` 0.15.0, `mypy-extensions` 1.1.0, `packaging` 26.3, `pathspec` 1.1.1,
`pluggy` 1.6.0, `pydantic-core` 2.46.5, `pygments` 2.21.0,
`python-dotenv` 1.2.3, `PyYAML` 6.0.3, `starlette` 1.7.0,
`truststore` 0.10.4, `typing-extensions` 4.16.0, `typing-inspection` 0.4.4,
`uvloop` 0.22.1, `watchfiles` 1.3.0, `websockets` 17.1

### Dependency decisions

1. **`httpx` → `httpx2`.** Starlette 1.7.0's `TestClient` emits a
   `StarletteDeprecationWarning` when backed by `httpx` and passes cleanly with
   `httpx2`. `httpx` was removed and the test suite re-run to confirm it is not
   needed transitively. Net dependency count is unchanged.
2. **No version was downgraded.** The only install failure in Phase 0-1 was a
   PyPI fetch timeout, resolved by raising `UV_HTTP_TIMEOUT` rather than by
   altering any version.

## Settled decisions

Each of these was deliberately left open until the phase that needed it, so the
choice was made with the real constraints in view rather than in advance:

| Decision | Phase | Choice |
| --- | --- | --- |
| LLM vendor and default model | 5 | Provider abstraction with OpenAI, Anthropic, and a deterministic fake; no vendor SDK reaches business logic. The default model is keyed by provider and resolved at build time, and the request body is shaped to the model — reasoning models reject `temperature` and take their ceiling as `max_completion_tokens` |
| Embedding provider and model | 6 | Local hashing embeddings by default; OpenAI optional. A model is never required just to run |
| Checkpoint backend | 21 | PostgreSQL through `langgraph-checkpoint-postgres`, so a run survives a restart |
| Authentication | 25 | JWT bearer tokens via PyJWT, with a trusted identity header allowed only in development |
| Event streaming | 24 | Server-sent events, no broker. The durable event log is the source of truth, so a dropped connection is recoverable rather than lost |
| Database tool target | 33 | A separate `DATABASE_TOOL_URL`; unset means the tool is not registered |
| Metrics and tracing | 27 | A dependency-free metrics registry with a Prometheus endpoint, and OpenTelemetry-format span tracing behind a protocol. No agent or collector is required to run |
| Control dashboard | 42 | Hand-written HTML, CSS, and JavaScript in the application, not a front-end framework. Three routes, no build step, and no dependency that needs its own supply chain |
| Container topology | 38 | A two-stage image plus compose; PostgreSQL and Redis as separate services with health gating, migrations applied at start-up |
| Quality evaluation | 36 | A labelled corpus in the repository with a measured rule-based baseline, so a score is always reported next to a floor |

## Still open

No phase remains open. Two things are unwritten rather than undecided, and both
are stated where they would otherwise be assumed:

- **The container image has never been built.** The files are complete and
  lint-checked; this host has no container runtime. CI builds it.
- **No vendor endpoint has ever been called.** Every suite, and every number in
  the README, comes from the deterministic fake provider. The adapters have now
  spoken real HTTP to a scripted loopback server, which covers the request and
  response contract — the endpoint path, the authentication header, the payload
  shape, the usage fields, and the error mapping. What that still cannot cover is
  vendor-specific behaviour: real rate-limit quirks, and anything only the live
  endpoint does. "The contract is tested over a socket" and "a real call has
  succeeded" are different claims, and only the first is true here.

  The model-specific payload rules are the sharpest example of the gap, and they
  are worth naming precisely. Which fields a model accepts is documented by the
  vendor and encoded here from that documentation: `temperature` and `max_tokens`
  are withheld from reasoning models because the vendor's own error text and its
  models page say those requests fail. That reasoning is sound, and it has been
  checked against a scripted server that echoes what it receives — but no live
  endpoint has confirmed it, and neither default model name has been sent
  anywhere. Treat the shipped names as a starting point to verify against the
  vendor's current lineup rather than as a tested configuration.
- ~~Frontend framework for the control dashboard — Phase 42.~~ Settled: no
  framework. See the table above.
