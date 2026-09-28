# LangGraph Multi-Agent System

A production-oriented multi-agent orchestration system built on LangGraph, with a
FastAPI interface, typed state, durable checkpointing, human-in-the-loop
approval, and provider-independent LLM access.

> **Build status: Phase 0-1 complete.** The application skeleton, tooling
> configuration, and a verified `GET /health` endpoint are in place. Every other
> subsystem listed below is planned, not implemented. See
> [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) for the phase-by-phase
> status. Nothing in this repository is claimed to be production-ready yet.

---

## Requirements

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | 3.12.x | The only interpreter verified on the development machine |
| `uv` | 0.12+ | Used for environment and dependency management |
| Git | 2.43+ | |
| Docker + Compose | — | **Not yet required.** Needed from Phase 38 onward |
| PostgreSQL | 16+ | **Not yet required.** Needed from Phase 3 onward |
| Redis | 7+ | **Not yet required.** Needed from Phase 4 onward |

## Setup

```bash
git clone <repository-url>
cd "Lang Graph"

# Create the virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the project with development tooling
uv pip install -e ".[dev]"

# Local configuration (gitignored)
cp .env.example .env
```

## Running

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness probe — returns `{"status": "ok"}` |
| `GET /docs` | Interactive OpenAPI documentation |
| `GET /openapi.json` | OpenAPI schema |

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}
```

## Development

```bash
ruff format .            # format
ruff check .             # lint
mypy app                 # type check (strict)
pytest                   # test suite
```

Run the full gate before every commit:

```bash
ruff format --check . && ruff check . && mypy app && pytest
```

## Project layout

```
app/
  main.py              # application factory and ASGI entry point
  api/routes/          # HTTP route modules
  core/                # configuration, logging, exceptions, security
  graph/               # LangGraph state, nodes, edges, routing, checkpoints
  agents/              # agent implementations
  tools/               # tools and the least-privilege registry
  memory/              # short-term, working, long-term, execution memory
  models/              # domain models
  schemas/             # Pydantic request/response/plan/event schemas
  services/            # LLM, embeddings, execution, memory, evaluation
  database/            # connection, ORM models, repositories
  observability/       # logging, tracing, metrics, events
tests/                 # unit, integration, graph, agent, tool, memory, api, evaluation
docs/                  # architecture and operational documentation
scripts/               # operational helper scripts
```

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Target architecture and control flow |
| [`docs/TECH_STACK.md`](docs/TECH_STACK.md) | Stack decisions and resolved versions |
| [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) | Phase plan and current status |

## Roadmap

Phases are implemented sequentially, and each phase must pass formatting,
linting, type checking, and tests before the next one begins.

- [x] **Phase 0** — environment discovery
- [x] **Phase 1** — base project, tooling, `GET /health`
- [ ] **Phase 2** — typed configuration
- [ ] **Phase 3** — PostgreSQL models, repositories, Alembic
- [ ] **Phase 4** — Redis cache
- [ ] **Phase 5** — LLM provider abstraction
- [ ] **Phase 6** — embedding provider abstraction
- [ ] **Phases 7-24** — state, events, tools, agents, graph, parallel execution,
      retries, critic, memory, checkpointing, approval, API, streaming
- [ ] **Phases 25-33** — authorization, rate limiting, observability, and the
      security hardening set
- [ ] **Phases 34-45** — testing, failure injection, evaluation, optimization,
      Docker, migrations, CI/CD, documentation, dashboard, final reviews
