# Development Plan

## Working agreement

The system is built in ordered phases. A phase is not complete until it passes
the full local gate, and no phase begins while the previous one is failing.

**Validation gate for every phase:**

```bash
ruff format .        # format
ruff check .         # lint
mypy app scripts     # strict type check (application + tooling)
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
| 2 | Typed configuration (Pydantic Settings) | **Complete** |
| 3 | PostgreSQL models, repositories, Alembic | Blocked — no PostgreSQL on host |
| 4 | Redis cache service | Blocked — no Redis on host |
| 5 | LLM provider abstraction | **Complete** |
| 6 | Embedding provider abstraction | **Complete** |
| 7 | Typed, serializable graph state | **Complete** |
| 8 | Structured execution events | **Complete** |
| 9 | Tool framework, registry, risk classification | **Complete** |
| 10 | Concrete tools | Not started |
| 11 | Tool security pipeline end to end | Partial — pipeline done, per-tool policies pending |
| 12 | Base agent contract | **Complete** |
| 13 | Planner agent | **Complete** |
| 14 | Structured intent routing | **Complete** |
| 15 | Specialist agents | Partial — research, coding, analysis, executor done; document agent pending |
| 16 | LangGraph orchestration graph | **Complete** |
| 17 | Bounded parallel dispatch | **Complete** |
| 18 | Failure classification and retry policy | **Complete** |
| 19 | Critic agent | **Complete** |
| 20 | Memory manager | Not started |
| 21 | Checkpointing and human approval | Partial — in-memory saver verified; durable PostgreSQL backend pending |
| 22 | Persist approval records | Not started |
| 23 | HTTP API | Not started |
| 24 | Execution-event streaming | Not started |
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

---

## Documentation, artwork, and licensing pass (post-Phase 1)

```
PHASE:              1 (addendum — documentation and branding)
STATUS:             Complete
FILES CREATED:      scripts/generate_assets.py, LICENSE,
                    docs/assets/{banner,logo,architecture,graph-flow,
                    tool-security,memory,roadmap}.png
FILES MODIFIED:     README.md, pyproject.toml, docs/DEVELOPMENT_PLAN.md
DEPENDENCIES:       pillow 12.3.0 (build-time only, not a runtime dependency);
                    pip-audit (development tooling, not declared in the manifest)
COMMANDS RUN:       python scripts/generate_assets.py
                    ruff format . && ruff check .
                    mypy app scripts
                    pytest
                    pip-audit
TESTS RUN:          tests/api/test_health.py (2 tests)
TEST RESULTS:       PASS — ruff clean (23 files formatted), mypy clean
                    (16 source files), 2 tests passed, pip-audit reports no
                    known vulnerabilities. All 7 README image references and
                    all local documentation links resolve.
ISSUES FOUND:       1. Ruff flagged 7 issues in the new asset script: two
                       S101 asserts, one unused local, four over-long string
                       literals.
                    2. MyPy flagged two real typing problems in the script:
                       a **kwargs dict passed to Image.save, and a union-typed
                       getextrema() return value.
                    3. The graph-flow diagram overflowed its canvas: the END
                       node was clipped, and the simple-request path drew
                       straight through the approval boxes.
ISSUES FIXED:       1. Replaced both asserts with real logic (a putdata-based
                       gradient with no pixel-access handle, and a documented
                       default font), removed the unused local, and split the
                       string literals.
                    2. Replaced the **kwargs with an explicit optimize flag,
                       and computed the blank-canvas guard from a histogram so
                       no union type is involved.
                    3. Enlarged the canvas and routed the simple path down a
                       dedicated right-hand channel clear of the approval
                       branch.
KNOWN LIMITATIONS:  The asset generator depends on Ubuntu system fonts at
                    /usr/share/fonts/truetype/ubuntu. On a host without them
                    the script fails loudly rather than rendering a fallback.
NEXT PHASE:         2 (typed configuration)
```

### Licensing decision

The project was initially declared MIT in `pyproject.toml`. It is now released
under a proprietary all-rights-reserved license (`LICENSE`), with copyright held
solely by the author. `pyproject.toml` points at the license file rather than
SPDX-identifying it.

Note that the repository itself is public while the license grants no rights:
the source is visible, but copying, modification, distribution, commercial use,
and use for model training are all expressly prohibited.

### Asset regeneration

The seven PNGs in `docs/assets/` are build artifacts of
`scripts/generate_assets.py`. Both the script and its output are committed, so
the diagrams can be reviewed as code and rebuilt deterministically.

---

## Phases 2-21 — the orchestration core

```
PHASE:              2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21
STATUS:             Complete (15 partial: document agent pending;
                    21 partial: durable backend pending)
FILES CREATED:      app/core/config.py, app/core/constants.py,
                    app/core/exceptions.py, app/services/llm.py,
                    app/services/embeddings.py, app/services/execution.py,
                    app/models/{tool,execution,agent,approval,memory}.py,
                    app/schemas/{plans,events}.py, app/graph/state.py,
                    app/graph/router.py, app/graph/nodes.py,
                    app/graph/builder.py, app/graph/checkpoints.py,
                    app/tools/base.py, app/tools/registry.py,
                    app/agents/base.py, app/agents/{planner,critic,
                    synthesizer,researcher,coder,analyst,executor}.py,
                    tests/unit/{test_config,test_exceptions,test_llm,
                    test_embeddings,test_state,test_plans,test_events,
                    test_tools}.py, tests/agents/test_agents.py,
                    tests/graph/{test_routing,test_graph_execution}.py
FILES MODIFIED:     pyproject.toml, docs/DEVELOPMENT_PLAN.md,
                    docs/TECH_STACK.md, README.md
DEPENDENCIES:       Added langgraph 1.2.12, pydantic-settings 2.15.0,
                    httpx 0.28.1 (runtime). langchain-core 1.6.5 arrives
                    transitively with langgraph and is not imported directly.
COMMANDS RUN:       uv pip install -e ".[dev,assets]"
                    ruff format . && ruff check .
                    mypy app scripts
                    pytest
TESTS RUN:          232 tests across unit, agents, and graph suites
TEST RESULTS:       PASS
                      ruff format --check ..... 65 files already formatted
                      ruff check .............. All checks passed
                      mypy app scripts ........ Success: no issues in 44 files
                      pytest .................. 232 passed
ISSUES FOUND:       1. `state.get(key) or default` treated a legitimate retry
                       ceiling of 0 as unset, letting a retry loop run that
                       should have been stopped.
                    2. The executor crashed the run when its target tool was
                       not registered.
                    3. Class methods `effective_risk()`/`requires_approval()`
                       read class attributes, so a test double that set the
                       access mode per instance silently bypassed the approval
                       gate and the test passed for the wrong reason.
                    4. `input_model`/`output_model` declared as instance
                       variables on a generic base could not be overridden by
                       a subclass class attribute under strict MyPy.
                    5. Two `str_replace` edits merged statements into
                       docstrings, producing syntax errors.
                    6. A blocking `time.sleep` in the fake provider serialised
                       the very work the concurrency test measured.
ISSUES FIXED:       All six. Notably (1) and (3) were real defects rather
                    than test noise: both would have let work proceed that the
                    configured limits should have stopped.
KNOWN LIMITATIONS:  - Checkpointing uses the in-memory saver. Interrupt and
                      resume are real, but a process restart loses in-flight
                      runs until the PostgreSQL saver is wired up.
                    - Concrete tools do not exist yet, so agents declare tool
                      allow-lists that a live registry cannot yet satisfy.
                      Calling `tools()` without registering them raises, which
                      is the intended failure mode.
                    - The document agent is not written.
                    - No HTTP API yet, so the graph is reachable only from
                      Python.
NEXT PHASE:         10 (concrete tools), then 20 (memory), 23 (API)
```

### Design decisions recorded

- **Structured output is implemented once, on the provider base class**, by
  instructing the model to emit JSON and validating against a Pydantic schema,
  with the validation error fed back for a bounded number of attempts. This
  keeps structured generation identical across vendors instead of depending on
  each vendor's tool-calling format.
- **The tool pipeline is enforced in `Tool.execute`**, not in each tool, so no
  individual tool can skip validation, the approval gate, or redaction.
- **Routing flags are derived, not trusted.** `requires_planning` comes from the
  route, and the approval route forces `requires_approval`. A model cannot
  misclassify a request by getting one field wrong.
- **Limits live on the state.** `iteration_limit` and `retry_limit` are copied
  into the state at run start so the conditional-edge routing functions are pure
  functions of state and can be tested without building a graph.
- **The critic never rewrites.** It reports; the orchestrator decides. A failed
  verdict retries while budget remains and then delivers anyway, with the
  unresolved criticism passed to the synthesizer as an explicit caveat.
