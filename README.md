<div align="center">

<img src="docs/assets/banner.png" alt="LangGraph Multi-Agent System" width="100%">

<br>

<img src="docs/assets/logo.png" alt="Logo" width="112">

# LANGGRAPH

**Multi-agent orchestration, engineered like infrastructure.**

Typed state · Durable checkpointing · Human-in-the-loop approval · Provider-independent LLMs

<br>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-FF6F61?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](#)
[![Redis](https://img.shields.io/badge/Redis-7+-DC382D?style=for-the-badge&logo=redis&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](#)

[![Ruff](https://img.shields.io/badge/Ruff-passing-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](#quality-gates)
[![MyPy](https://img.shields.io/badge/MyPy-strict-2A6DB2?style=for-the-badge)](#quality-gates)
[![Tests](https://img.shields.io/badge/tests-2%20passing-brightgreen?style=for-the-badge)](#quality-gates)

[![Status](https://img.shields.io/badge/status-phase%200--1-yellow?style=for-the-badge)](#build-status)
[![License](https://img.shields.io/badge/license-proprietary-red?style=for-the-badge)](#license)

</div>

---

## Build status

> [!WARNING]
> **This repository is at Phase 0-1 of 45.** Implemented and verified: the project
> skeleton, the dependency manifest, the quality gate, and `GET /health`.
>
> Everything else documented below — the graph, agents, tools, memory,
> checkpointing, approval workflow, authentication, and Docker stack — is
> **designed but not yet written**. Each capability is marked `planned` in the
> tables that follow. Nothing here is claimed to be production-ready.

**Verified on this machine**

| Gate | Command | Result |
| :--- | :--- | :--- |
| Formatting | `ruff format --check .` | 22 files already formatted |
| Linting | `ruff check .` | All checks passed |
| Types | `mypy app scripts` | Success: no issues in 16 source files |
| Tests | `pytest` | 2 passed |
| Dependencies | `pip-audit` | No known vulnerabilities found |
| Startup | `uvicorn app.main:app` | Application startup complete |
| Endpoint | `curl /health` | `200 {"status":"ok"}` |

**Status legend used throughout:** ✅ implemented and verified · 🔷 designed, not yet written · ⛔ blocked by a missing dependency on the host

---

## Table of contents

- [What this is](#what-this-is)
- [Architecture](#architecture)
- [Execution graph](#execution-graph)
- [How it compares](#how-it-compares)
- [Technology stack](#technology-stack)
- [Tool security model](#tool-security-model)
- [Memory architecture](#memory-architecture)
- [Failure and retry policy](#failure-and-retry-policy)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [API surface](#api-surface)
- [Configuration](#configuration)
- [Quality gates](#quality-gates)
- [Testing strategy](#testing-strategy)
- [Security model](#security-model)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [License](#license)

---

## What this is

A multi-agent system that routes a request through an explicit, typed graph
rather than an unsupervised chain of prompts. A router decides whether the task
is simple or complex; complex tasks are planned, dispatched to specialised
agents that may run in parallel, aggregated, verified by a critic, and only then
synthesised into a final answer. Actions that carry risk are gated behind human
approval, and the entire run is checkpointed so it can be interrupted and
resumed.

The design commitments that shape everything else:

| Commitment | Why it matters |
| :--- | :--- |
| **Explicit graph, not free-running agents** | Control flow is inspectable and testable. There is no "hope it terminates" path. |
| **Every loop is bounded** | Iterations, tool calls, wall-clock time, and retries all have configured ceilings. |
| **Typed, serializable state** | State must survive checkpointing, so `dict[str, Any]` everywhere is ruled out. |
| **Provider-independent LLM access** | Agents depend on an interface, so no vendor SDK leaks into business logic. |
| **Least-privilege tooling** | An agent receives only the tools it is explicitly authorised to use. |
| **External content is untrusted** | Retrieved web pages and documents are data, never instructions. |
| **No hidden reasoning exposed** | Clients get status, tool activity, and safe summaries — never chain-of-thought. |

---

## Architecture

<img src="docs/assets/architecture.png" alt="Layered system architecture" width="100%">

Five layers, each depending only on the abstractions beneath it. The API layer
holds no business logic; the agent layer never touches a database or the graph
runtime directly.

| Layer | Responsibility | Status |
| :--- | :--- | :--- |
| **Edge / API** | HTTP surface, request validation, middleware, route handlers. Contains no business logic. | 🔷 |
| **Orchestration** | Task lifecycle, LangGraph runtime, intent routing, planning, dispatch, criticism, synthesis. | 🔷 |
| **Agents** | Research, coding, analysis, document, and executor agents behind a base contract. | 🔷 |
| **Capability** | Tool registry, memory manager, checkpoint store, LLM and embedding abstractions. | 🔷 |
| **Infrastructure** | PostgreSQL for durability, Redis for cache and rate limiting, event log for audit. | ⛔ |

### Directory-to-layer mapping

| Path | Layer | Phase |
| :--- | :--- | :--- |
| `app/api/` | Edge | 23, 25, 26 |
| `app/graph/` | Orchestration | 7, 16, 21 |
| `app/agents/` | Agents | 12-15, 19 |
| `app/tools/` | Capability | 9-11 |
| `app/memory/` | Capability | 20 |
| `app/services/` | Capability | 4-6, 36 |
| `app/database/` | Infrastructure | 3 |
| `app/observability/` | Cross-cutting | 27 |
| `app/core/` | Cross-cutting | 2 |

---

## Execution graph

<img src="docs/assets/graph-flow.png" alt="LangGraph execution flow" width="100%">

Every branch is decided by a conditional edge over typed state — never by
string-matching free-form model output.

| Route | Trigger | Path |
| :--- | :--- | :--- |
| `DIRECT` | Trivial request, no tools needed | Router → direct response → end |
| `RESEARCH` | Fact-finding with sources | Router → planner → research agent → critic → synthesizer |
| `CODING` | Code generation or debugging | Router → planner → coding agent → critic → synthesizer |
| `DATA_ANALYSIS` | Computation over data | Router → planner → analysis agent → critic → synthesizer |
| `DOCUMENT` | Extraction from documents | Router → planner → document agent → critic → synthesizer |
| `MULTI_AGENT` | Spans several capabilities | Router → planner → parallel agents → aggregator → critic → synthesizer |
| `HUMAN_APPROVAL` | Risky or destructive action | Risk check → interrupt → human decision → resume |

**Critic outcome is advisory, not authoritative.** The critic reports; the
orchestrator decides whether to pass, retry, replan, or fail. The critic never
silently rewrites an agent's output.

---

## How it compares

### Against an unsupervised agent loop

The common alternative is a single agent looping until it decides it is done.
That is simpler, and for narrow tasks it is fine. The trade-offs are concrete:

| Dimension | Unsupervised agent loop | This system | Status |
| :--- | :--- | :--- | :--- |
| Control flow | Emergent, decided by the model at runtime | Explicit graph with declared edges | 🔷 |
| Termination | Model decides; can loop indefinitely | Hard iteration, timeout, and tool-call ceilings | 🔷 |
| Crash recovery | Run is lost | Durable checkpoints; resume from last node | 🔷 |
| Verification | None by default | Independent critic with structured findings | 🔷 |
| Risky actions | Executed if the model chooses | Risk classification gates execution | 🔷 |
| Auditability | Prompt transcript only | Structured events per node, agent, and tool call | 🔷 |
| Testability | Requires live model calls | Deterministic fake providers; graph runs offline | 🔷 |
| Cost control | Unbounded token spend | Per-task token budgets and early termination | 🔷 |
| Vendor lock-in | Often coupled to one SDK | Provider abstraction behind a typed interface | 🔷 |

The deliberate cost of this design is more moving parts. The trade is made
because unattended, unbounded, unverifiable execution is unacceptable for
anything that can write files or touch a database.

### Against other orchestration approaches

| Approach | Model | Strength | Why this project chose otherwise |
| :--- | :--- | :--- | :--- |
| **LangGraph** ✅ chosen | Explicit state graph | Durable checkpointing and interrupt-based human-in-the-loop are first-class | — |
| Role-based crews | Agents with assigned roles collaborating | Fast to prototype | Control flow stays implicit; harder to bound and audit |
| Conversation-centric multi-agent | Agents talking in a shared message loop | Natural for open-ended dialogue | Termination and cost are harder to bound deterministically |
| Hand-rolled orchestration | Custom control flow | Total control | Re-implements checkpointing, interrupts, and state persistence |

> Comparison rows are high-level and reflect documented capabilities of each
> approach, not benchmarks. No performance comparison has been measured — this
> project has no working implementation to benchmark yet.

---

## Technology stack

Every dependency has to earn its place; nothing is included because it is
fashionable.

| Concern | Choice | Rationale | Status |
| :--- | :--- | :--- | :--- |
| Language | Python 3.12 | Only interpreter on the target machine; fully supported by the stack | ✅ |
| Dependency management | `uv` | Present on the host; fast resolver | ✅ |
| API framework | FastAPI | Async-native, Pydantic-validated, generates OpenAPI | ✅ |
| ASGI server | Uvicorn | Standard FastAPI companion | ✅ |
| Validation | Pydantic v2 | Runtime validation and the schema source for structured LLM output | ✅ |
| Configuration | Pydantic Settings | Strongly typed config with `.env` loading | 🔷 |
| Orchestration | LangGraph | The graph runtime this architecture is built around | 🔷 |
| LLM integration | Provider abstraction over LangChain interfaces | Keeps agents vendor-neutral | 🔷 |
| Database | PostgreSQL + SQLAlchemy + Alembic | Durable tasks, approvals, memory, checkpoints | ⛔ |
| Cache / coordination | Redis | Rate limiting and short-lived coordination only | ⛔ |
| Testing | pytest + pytest-asyncio | Async support for graph and API tests | ✅ |
| Linting / formatting | Ruff | One tool for both | ✅ |
| Type checking | MyPy (strict) | A typed codebase is an explicit requirement | ✅ |
| Containerisation | Docker + Compose | Reproducible local stack | ⛔ |

### Resolved versions

Direct dependencies are pinned exactly; the full transitive set is recorded in
[`docs/TECH_STACK.md`](docs/TECH_STACK.md).

| Package | Version | Scope |
| :--- | :--- | :--- |
| `fastapi` | 0.141.1 | runtime |
| `pydantic` | 2.13.5 | runtime |
| `uvicorn[standard]` | 0.54.0 | runtime |
| `httpx2` | 2.13.1 | dev — test client transport |
| `mypy` | 2.3.1 | dev |
| `pytest` | 9.1.1 | dev |
| `pytest-asyncio` | 1.4.0 | dev |
| `ruff` | 0.16.9 | dev |

> **Decision of note:** Starlette 1.7 deprecates `httpx` in favour of `httpx2`
> for its test client. This project uses `httpx2`, and `httpx` was removed after
> confirming the suite passes without it. No dependency was ever downgraded to
> resolve a conflict — the only install failure so far was a network timeout.

---

## Tool security model

<img src="docs/assets/tool-security.png" alt="Tool security pipeline and risk ladder" width="100%">

Every tool call passes the same pipeline. Permission, risk, and approval checks
all happen *before* execution, and every call emits an audit event.

### Risk levels and execution policy

| Level | Example operations | Execution policy | Status |
| :--- | :--- | :--- | :--- |
| **LOW** | Read a file, search the web | Auto-execute | 🔷 |
| **MEDIUM** | Write a file, open an issue | Auto-execute, audited | 🔷 |
| **HIGH** | Execute code, write to the database | Human approval required | 🔷 |
| **CRITICAL** | Delete data, drop a table | Human approval required | 🔷 |

### Per-tool restrictions

| Tool | Default mode | Additional restriction | Status |
| :--- | :--- | :--- | :--- |
| `WebSearchTool` | Read | URL validation; blocks private, loopback, and metadata addresses | 🔷 |
| `FilesystemTool` | Read | Confined to an allow-listed root; traversal and symlink escapes rejected; size cap | 🔷 |
| `PythonExecutionTool` | **Disabled** | Must run in an isolated sandbox. Disabled rather than faked | 🔷 |
| `DatabaseTool` | Read-only | Writes need authorisation; destructive statements need approval | 🔷 |
| `GitHubTool` | Read | Writes need authorisation; tokens never exposed to the model | 🔷 |

> **On sandboxing:** arbitrary Python execution is the highest-risk capability in
> any agent system. This host has no container runtime, so
> `PYTHON_EXECUTION_ENABLED` defaults to `false`. The feature stays off until a
> real isolation boundary exists, rather than shipping something that only
> pretends to be safe.

---

## Memory architecture

<img src="docs/assets/memory.png" alt="Memory architecture" width="100%">

| Tier | Contents | Storage | Lifetime | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Short-term** | Current conversation context | In-process buffer | Conversation | 🔷 |
| **Working** | State of the task in flight | Graph state | Task | 🔷 |
| **Long-term semantic** | Durable facts worth keeping | PostgreSQL + vectors | Indefinite | 🔷 |
| **Execution** | What previous runs did | Execution records | Indefinite | 🔷 |

All four sit behind one manager (`retrieve`, `store`, `update`, `summarize`,
`delete`). Relevance is evaluated *before* promotion to durable storage — not
every message is worth remembering, and memory pollution is a real failure mode.

---

## Failure and retry policy

Failures are classified before any retry decision is made, so a permanent error
is never retried in a loop.

| Classification | Retried? | Backoff | Status |
| :--- | :--- | :--- | :--- |
| `TRANSIENT` | Yes | Exponential | 🔷 |
| `RATE_LIMIT` | Yes | Exponential, longer base | 🔷 |
| `TIMEOUT` | Yes | Exponential | 🔷 |
| `TOOL_FAILURE` | Yes, if idempotent | Exponential | 🔷 |
| `MODEL_FAILURE` | Yes | Exponential | 🔷 |
| `VALIDATION` | No | — | 🔷 |
| `AUTHENTICATION` | No | — | 🔷 |
| `DATABASE_FAILURE` | Yes, limited | Exponential | 🔷 |
| `PERMANENT` | No | — | 🔷 |
| `UNKNOWN` | Once | Fixed | 🔷 |

**Destructive actions are never retried automatically**, regardless of
classification.

---

## Project layout

```
.
├── app/
│   ├── main.py                 # application factory + ASGI entry point
│   ├── api/
│   │   ├── routes/             # health, chat, tasks, agents, approvals
│   │   ├── dependencies.py     # shared FastAPI dependencies
│   │   └── middleware.py       # auth, rate limiting, request context
│   ├── core/                   # config, logging, exceptions, security, constants
│   ├── graph/                  # state, nodes, edges, router, checkpoints, builder
│   ├── agents/                 # base, planner, researcher, coder, analyst,
│   │                           # executor, critic, synthesizer
│   ├── tools/                  # base, registry, web_search, filesystem,
│   │                           # python_executor, database, github
│   ├── memory/                 # short_term, long_term, semantic, manager
│   ├── models/                 # task, execution, agent, tool, approval, memory
│   ├── schemas/                # requests, responses, plans, events
│   ├── services/               # llm, embeddings, execution, memory, evaluation
│   ├── database/               # connection, models, repositories
│   └── observability/          # tracing, metrics, events
├── tests/
│   ├── unit/  integration/  graph/  agents/
│   ├── tools/  memory/  api/  evaluation/
├── scripts/                    # generate_assets.py, operational helpers
├── docs/
│   ├── assets/                 # generated diagrams (PNG)
│   ├── ARCHITECTURE.md
│   ├── TECH_STACK.md
│   └── DEVELOPMENT_PLAN.md
├── pyproject.toml
├── .env.example
├── LICENSE
└── README.md
```

Packages marked in the tree but not yet implemented contain only a docstring —
they exist to lock in the layout. There are no stub or fake implementations.

---

## Getting started

### Prerequisites

| Requirement | Version | Needed now? |
| :--- | :--- | :--- |
| Python | 3.12.x | Yes |
| `uv` | 0.12+ | Yes |
| Git | 2.43+ | Yes |
| PostgreSQL | 16+ | No — from Phase 3 |
| Redis | 7+ | No — from Phase 4 |
| Docker + Compose | current | No — from Phase 38 |

### Install

```bash
git clone https://github.com/officialarghya29/LangGraph.git
cd LangGraph

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
```

### Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

```console
$ curl -s http://127.0.0.1:8000/health
{"status":"ok"}
```

| URL | Purpose |
| :--- | :--- |
| `http://127.0.0.1:8000/health` | Liveness probe |
| `http://127.0.0.1:8000/docs` | Interactive OpenAPI documentation |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI schema |

---

## API surface

| Method | Path | Purpose | Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness. No dependency checks | ✅ |
| `GET` | `/ready` | Readiness, including dependencies | 🔷 |
| `POST` | `/api/v1/chat` | Conversational entry point | 🔷 |
| `POST` | `/api/v1/tasks` | Create a task | 🔷 |
| `GET` | `/api/v1/tasks/{task_id}` | Fetch a task | 🔷 |
| `GET` | `/api/v1/tasks/{task_id}/status` | Execution status | 🔷 |
| `POST` | `/api/v1/tasks/{task_id}/approve` | Approve a gated action | 🔷 |
| `POST` | `/api/v1/tasks/{task_id}/reject` | Reject a gated action | 🔷 |
| `POST` | `/api/v1/tasks/{task_id}/cancel` | Cancel a running task | 🔷 |
| `GET` | `/api/v1/agents` | List available agents | 🔷 |
| `GET` | `/api/v1/tools` | List available tools | 🔷 |

Streaming exposes execution events only — `task_started`, `planning`,
`agent_started`, `tool_completed`, `verification_completed`,
`approval_required`, `task_completed`, `task_failed`. Never chain-of-thought,
prompts, or credentials.

---

## Configuration

All configuration flows through one typed settings layer. Application code never
reads environment variables or secrets directly.

| Variable | Default | Purpose | Status |
| :--- | :--- | :--- | :--- |
| `APP_NAME` | `langgraph-multi-agent` | Service identity | 🔷 |
| `APP_ENV` | `development` | `development` / `staging` / `production` | 🔷 |
| `DEBUG` | `false` | Verbose error output | 🔷 |
| `LOG_LEVEL` | `INFO` | Logging threshold | 🔷 |
| `LLM_PROVIDER` | `openai` | Provider selection | 🔷 |
| `LLM_MODEL` | — | Model name, never hardcoded in agents | 🔷 |
| `LLM_API_KEY` | — | Provider credential | 🔷 |
| `EMBEDDING_PROVIDER` | `local` | Embedding backend | 🔷 |
| `DATABASE_URL` | local Postgres | Durable storage | 🔷 |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache and rate limiting | 🔷 |
| `MAX_AGENT_ITERATIONS` | `10` | Loop ceiling | 🔷 |
| `MAX_TOOL_CALLS` | `25` | Tool-call ceiling | 🔷 |
| `MAX_EXECUTION_TIME` | `300` | Wall-clock ceiling (seconds) | 🔷 |
| `MAX_PARALLEL_TASKS` | `4` | Concurrency ceiling | 🔷 |
| `MAX_RETRIES` | `3` | Retry ceiling | 🔷 |
| `AUTH_ENABLED` | `true` | Authentication toggle | 🔷 |
| `PYTHON_EXECUTION_ENABLED` | `false` | Sandboxed execution; off until isolation exists | 🔷 |
| `GITHUB_TOOL_ALLOW_WRITES` | `false` | GitHub write toggle | 🔷 |

See [`.env.example`](.env.example) for the complete annotated list.

---

## Quality gates

```bash
ruff format .            # format
ruff check .             # lint
mypy app scripts         # strict type check
pytest                   # tests
```

The full gate before any commit:

```bash
ruff format --check . && ruff check . && mypy app scripts && pytest
```

Dependency vulnerabilities are audited separately, since it needs network
access:

```bash
pip-audit
```

MyPy runs in `strict` mode. Ruff enables `E`, `W`, `F`, `I`, `N`, `UP`, `B`,
`C4`, `SIM`, `ASYNC`, `S`, `RUF`, and `D` — the `S` (bandit) rules are on
deliberately, so common security mistakes fail the build rather than the review.

A phase is not complete until this gate passes. No phase begins while the
previous one is failing.

---

## Testing strategy

| Layer | Scope | Approach |
| :--- | :--- | :--- |
| **Unit** | Config, state, schemas, router, planner, agents, registry, risk classifier, memory | Pure functions, no network |
| **Integration** | Database, Redis, checkpointing, graph execution, tool execution | Real services where available |
| **Graph** | Simple, complex, parallel, critic pass/fail, retry, replan, approval, rejection, resume, recovery | Deterministic fake providers |
| **API** | Chat, task creation, status, approve, reject, cancel, health | FastAPI test client |
| **Security** | Path traversal, prompt injection, unauthorised tools, cross-user access, SSRF, unsafe SQL, secret leakage | Adversarial cases |
| **Evaluation** | Routing accuracy, planning accuracy, tool selection, completion, verification, response quality, citations | Rule-based scoring, not an LLM judge alone |

All LLM calls are replaced by deterministic fakes in unit and graph tests, so the
suite runs offline and produces stable results.

---

## Security model

| Threat | Control | Status |
| :--- | :--- | :--- |
| Path traversal | Allow-listed roots, normalised paths, symlink escape checks | 🔷 |
| Prompt injection | Retrieved content is data, never instructions; never overrides system policy | 🔷 |
| SSRF | URL validation; blocks loopback, private ranges, and metadata endpoints | 🔷 |
| Unauthorised tool use | Per-agent allow-lists; only explicitly granted tools are visible | 🔷 |
| Cross-user data access | Ownership enforced on tasks, conversations, memory, approvals | 🔷 |
| Unsafe SQL | Read-only by default; no model-authored administrative statements | 🔷 |
| Secret leakage | Secrets only from config; never logged, returned, or shown to the model | 🔷 |
| Unbounded execution | Iteration, tool-call, time, and retry ceilings | 🔷 |
| Unsafe code execution | Disabled until a genuine isolation boundary exists | ✅ |

---

## Roadmap

<img src="docs/assets/roadmap.png" alt="Build roadmap" width="100%">

<details open>
<summary><b>Phase progress</b></summary>

- [x] **Phase 0** — environment discovery
- [x] **Phase 1** — base project, tooling, `GET /health`
- [ ] **Phase 2** — typed configuration (Pydantic Settings)
- [ ] **Phase 3** — PostgreSQL models, repositories, Alembic ⛔
- [ ] **Phase 4** — Redis cache service ⛔
- [ ] **Phase 5** — LLM provider abstraction
- [ ] **Phase 6** — embedding provider abstraction
- [ ] **Phases 7-24** — state, events, tools, agents, graph, parallel execution, retries, critic, memory, checkpointing, human approval, API, streaming
- [ ] **Phases 25-33** — authorization, rate limiting, observability, prompt injection, SSRF, filesystem, execution, database and GitHub hardening
- [ ] **Phases 34-37** — testing, failure injection, evaluation, efficiency optimisation
- [ ] **Phases 38-45** — Docker, migrations, CI/CD, documentation, control dashboard, final security/performance/architecture reviews

</details>

> [!IMPORTANT]
> **Phases 3, 4, 38, and 39 are blocked.** PostgreSQL, Redis, and Docker are not
> installed on the development host. That code will not be written blind — an
> unverifiable database or container implementation would have to ship untested,
> which is worse than shipping nothing.

---

## Documentation

| Document | Contents |
| :--- | :--- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layer responsibilities, control flow, key decisions |
| [`docs/TECH_STACK.md`](docs/TECH_STACK.md) | Environment discovery, stack rationale, resolved versions |
| [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) | Phase plan, validation protocol, phase reports |
| [`scripts/generate_assets.py`](scripts/generate_assets.py) | Regenerates every diagram in this README |

Diagrams are generated from code rather than checked in as opaque binaries.
To rebuild them:

```bash
uv pip install -e ".[assets]"
python scripts/generate_assets.py
```

---

## License

**Proprietary — all rights reserved.**

Copyright © 2026 **Arghya Bose**. All rights, title, and interest in this
software, including all source code, documentation, architecture, designs, and
assets, belong exclusively to the author.

No license or permission of any kind is granted. Copying, modifying,
distributing, sublicensing, commercial use, and use for training machine
learning models are all expressly prohibited without prior written permission.

See [LICENSE](LICENSE) for the full terms.

---

<div align="center">

**Built as infrastructure, not as a demo.**

<sub>Copyright © 2026 Arghya Bose. All rights reserved.</sub>

</div>
