# Development Plan

## Working agreement

The system is built in ordered phases. A phase is not complete until it passes
the full local gate, and no phase begins while the previous one is failing.

**Validation gate for every phase:**

```bash
ruff format .        # format
ruff check .         # lint
mypy app             # strict type check
pytest               # tests
```

Plus, where applicable: verification that the application starts, that imports
resolve, and that the endpoints introduced by the phase respond correctly.

**Phase report template** (recorded in this file as phases complete):

```
PHASE:
STATUS:
FILES CREATED:
FILES MODIFIED:
DEPENDENCIES:
COMMANDS RUN:
TESTS RUN:
TEST RESULTS:
ISSUES FOUND:
ISSUES FIXED:
KNOWN LIMITATIONS:
NEXT PHASE:
```

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Environment discovery | **Complete** |
| 1 | Base project, tooling, `GET /health` | **Complete** |
| 2 | Typed configuration (Pydantic Settings) | Not started |
| 3 | PostgreSQL models, repositories, Alembic | Blocked — no PostgreSQL on host |
| 4 | Redis cache service | Blocked — no Redis on host |
| 5 | LLM provider abstraction | Not started |
| 6 | Embedding provider abstraction | Not started |
| 7-24 | State, events, tools, agents, graph, parallelism, retries, critic, memory, checkpointing, approval, API, streaming | Not started |
| 25-33 | Authorization, rate limiting, observability, security hardening | Not started |
| 34-37 | Testing, failure injection, evaluation, optimization | Not started |
| 38-39 | Docker, migration verification | Blocked — no Docker on host |
| 40-45 | CI/CD, documentation, dashboard, final reviews | Not started |

**Blocked phases:** 3, 4, 38, 39 need services that are not installed. They stay
blocked until PostgreSQL, Redis, and Docker are available; the code for them
will not be written blind, because unverifiable database and container work
would have to be shipped untested.

---

## Phase 0 — Environment discovery

```
PHASE:              0
STATUS:             Complete
FILES CREATED:      docs/TECH_STACK.md, docs/DEVELOPMENT_PLAN.md
FILES MODIFIED:     none
DEPENDENCIES:       none installed
COMMANDS RUN:       pwd, uname -a, ls -la, python3 --version, pip --version,
                    venv check, uv/poetry probe, git --version, git config,
                    docker --version, docker compose version, docker info,
                    psql --version, redis-cli --version, curl pypi probe
TESTS RUN:          none (discovery only)
TEST RESULTS:       n/a
ISSUES FOUND:       Docker, PostgreSQL, and Redis are not installed. PyPI is
                    reachable but slow, causing one install timeout.
ISSUES FIXED:       Raised the uv HTTP timeout via UV_HTTP_TIMEOUT=120.
KNOWN LIMITATIONS:  Phases 3, 4, 38 and 39 cannot be validated in this
                    environment.
NEXT PHASE:         1
```

Findings are recorded in [TECH_STACK.md](TECH_STACK.md).

## Phase 1 — Base project, tooling, health endpoint

```
PHASE:              1
STATUS:             Complete
FILES CREATED:      pyproject.toml, .gitignore, .env.example, README.md,
                    app/__init__.py, app/main.py, app/api/__init__.py,
                    app/api/routes/__init__.py, app/api/routes/health.py,
                    app/core/__init__.py, app/graph/__init__.py,
                    app/agents/__init__.py, app/tools/__init__.py,
                    app/memory/__init__.py, app/models/__init__.py,
                    app/schemas/__init__.py, app/services/__init__.py,
                    app/database/__init__.py, app/observability/__init__.py,
                    tests/__init__.py, tests/api/__init__.py,
                    tests/api/test_health.py, docs/ARCHITECTURE.md
FILES MODIFIED:     docs/DEVELOPMENT_PLAN.md
DEPENDENCIES:       fastapi 0.141.1, pydantic 2.13.5, uvicorn[standard] 0.54.0
                    (+ httpx2 2.13.1, mypy 2.3.1, pytest 9.1.1,
                    pytest-asyncio 1.4.0, ruff 0.16.9 for development).
                    Full transitive set in TECH_STACK.md
COMMANDS RUN:       uv venv --python 3.12 .venv
                    uv pip install -e ".[dev]"
                    ruff format .
                    ruff check .
                    mypy app
                    pytest -q
                    uvicorn app.main:app --host 127.0.0.1 --port 8123
                    curl /health, curl /openapi.json
TESTS RUN:          tests/api/test_health.py (2 tests)
TEST RESULTS:       PASS
                      ruff format --check ....... 22 files already formatted
                      ruff check ................ All checks passed
                      mypy app .................. Success: no issues found
                                                  in 15 source files
                      pytest .................... 2 passed
                      uvicorn startup ........... "Application startup complete"
                      GET /health ............... HTTP 200 {"status":"ok"}
                      GET /openapi.json ......... 200, paths: ['/health']
ISSUES FOUND:       1. PyPI fetch timeout on `watchfiles` during install.
                    2. Editable build failed on missing README.md.
                    3. Starlette 1.7 deprecated `httpx` in favour of `httpx2`.
                    4. Ruff flagged unsorted imports in tests/api/test_health.py.
ISSUES FIXED:       All four, without changing any dependency version except
                    the httpx -> httpx2 swap described below.
KNOWN LIMITATIONS:  No Docker, PostgreSQL, or Redis, so Phases 3, 4, 38 and 39
                    remain blocked. Structure-only packages are empty by
                    design: no stub or fake implementations are present.
NEXT PHASE:         2 (typed configuration)
```

### Issues encountered and resolved

1. **`uv pip install` fetch timeout on `watchfiles`.** Not a version conflict —
   a network timeout against PyPI. Resolved by raising `UV_HTTP_TIMEOUT` to 120
   seconds rather than changing any dependency version.
2. **Editable build failed: `Readme file does not exist: README.md`.** The
   package metadata declared `readme = "README.md"` before the file existed.
   Resolved by creating `README.md`.
3. **`StarletteDeprecationWarning`: `httpx` deprecated in favour of `httpx2`.**
   Verified `httpx2` 2.13.1 is published, installed it, and confirmed the
   warning disappears. Then uninstalled `httpx` and re-ran the suite to prove it
   is not needed transitively. `httpx` was replaced by `httpx2` in the dev extra.
4. **Ruff `I001` import ordering in the test module.** Fixed with
   `ruff check --fix .`, then re-verified.

### Deferred cleanup

The empty structural packages under `app/` (`core`, `graph`, `agents`, `tools`,
`memory`, `models`, `schemas`, `services`, `database`, `observability`) contain
only a package docstring. They are placeholders for the directory layout, not
stub implementations, and Phase 16 onward fills them in.
