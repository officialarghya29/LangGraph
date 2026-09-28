# syntax=docker/dockerfile:1
#
# Two stages, because the build tools needed to install the dependency tree are
# not needed to run it. The runtime image carries the virtual environment only —
# no compilers, no package indexes, and no build cache that could be used to
# install something after the fact.
#
# NOTE: this image has not been built or run. This host has no container runtime,
# so everything here is written and lint-checked but unverified. Treat it as a
# starting point that must be built once before it is trusted.

# --------------------------------------------------------------------------- #
# Build stage
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

# Pinned by version rather than by ``latest``: a dependency resolution that
# changes without a commit is not reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Only the files needed to resolve and install dependencies, so a change to the
# application code does not invalidate the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY app ./app

RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache .

# --------------------------------------------------------------------------- #
# Runtime stage
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="langgraph-multi-agent" \
      org.opencontainers.image.description="LangGraph multi-agent orchestration system" \
      org.opencontainers.image.licenses="Proprietary"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production

# A non-root account. The application writes nothing to its own filesystem, so
# there is no reason for it to be able to.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser alembic.ini pyproject.toml ./

USER appuser

EXPOSE 8000

# Reads the readiness endpoint, which checks PostgreSQL and Redis rather than
# only the process, so a container that is up but cannot reach its dependencies
# is reported unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
