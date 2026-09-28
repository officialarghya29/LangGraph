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
| Configuration | Pydantic Settings | Strongly typed config with `.env` support *(Phase 2)* |
| Orchestration | LangGraph | The graph runtime the architecture is built around *(Phase 16)* |
| LLM integration | LangChain core interfaces, wrapped in an in-house provider abstraction | Keeps agents vendor-independent *(Phase 5)* |
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
| `pydantic` | 2.13.5 | Runtime validation; also the schema source for LLM structured output |
| `uvicorn[standard]` | 0.54.0 | ASGI server; extras add `httptools`, `uvloop`, `watchfiles`, `websockets`, `python-dotenv`, `PyYAML` |

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

## Deferred decisions

The following are deliberately **not** chosen yet. Each is decided in the phase
where it is first needed, so the decision is made with the real constraints in
view:

- LLM vendor and default model — Phase 5.
- Embedding provider and model — Phase 6.
- Checkpoint persistence backend and serialization format — Phase 21.
- Authentication mechanism (JWT vs. session vs. external provider) — Phase 25.
- Message broker, if any, for execution-event streaming — Phase 24.
- Frontend framework for the control dashboard — Phase 42.
