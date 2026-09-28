#!/usr/bin/env bash
#
# Bring up the local PostgreSQL and Redis instances used to verify this project.
#
# The development host has no root access and no container runtime, so the
# services are conda-forge builds installed under .services/runtime and run as
# the current user. This is a real PostgreSQL and a real Redis, not a stand-in,
# which is what makes the migration, cache, and checkpoint tests meaningful.
#
# Everything lives under .services/, which is git-ignored: no binaries, data
# directories, logs, or sockets are committed.
#
# Usage:
#   scripts/dev_services.sh up       # initialise (once) and start both services
#   scripts/dev_services.sh down     # stop both services
#   scripts/dev_services.sh status   # report what is running
#   scripts/dev_services.sh reset    # stop, delete all data, and start clean
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICES="${ROOT}/.services"
RUNTIME="${SERVICES}/runtime"
BIN="${RUNTIME}/bin"

PGDATA="${SERVICES}/pgdata"
PGRUN="${SERVICES}/run"
REDISDATA="${SERVICES}/redisdata"
LOGS="${SERVICES}/logs"

PGPORT="${PGPORT:-55432}"
REDISPORT="${REDISPORT:-56379}"

# Local development credentials. Not secrets: this instance listens only on the
# loopback interface and is never reachable from outside the machine. Any real
# deployment supplies its own values through the environment.
PGUSER="${PGUSER:-langgraph}"
PGPASSWORD="${PGPASSWORD:-langgraph}"
PGDATABASE="${PGDATABASE:-langgraph}"

die() { echo "error: $*" >&2; exit 1; }

require_runtime() {
  [[ -x "${BIN}/postgres" ]] || die "PostgreSQL is not installed. Run: scripts/dev_services.sh setup"
  [[ -x "${BIN}/redis-server" ]] || die "Redis is not installed. Run: scripts/dev_services.sh setup"
}

setup() {
  if [[ -x "${BIN}/postgres" && -x "${BIN}/redis-server" ]]; then
    echo "runtime already present at ${RUNTIME}"
    return 0
  fi

  command -v curl >/dev/null || die "curl is required to download micromamba"
  mkdir -p "${SERVICES}/micromamba"
  echo "downloading micromamba..."
  curl -sL -o /tmp/mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
  tar -xjf /tmp/mm.tar.bz2 -C "${SERVICES}/micromamba"
  chmod +x "${SERVICES}/micromamba/bin/micromamba"
  rm -f /tmp/mm.tar.bz2

  echo "installing postgresql and redis-server from conda-forge..."
  MAMBA_ROOT_PREFIX="${SERVICES}/mamba" MAMBA_NO_BANNER=1 \
    "${SERVICES}/micromamba/bin/micromamba" create -y -p "${RUNTIME}" \
    -c conda-forge postgresql redis-server >/dev/null

  echo "runtime installed at ${RUNTIME}"
}

init_postgres() {
  [[ -d "${PGDATA}/base" ]] && return 0

  mkdir -p "${PGRUN}" "${LOGS}"
  echo "initialising the PostgreSQL cluster..."
  # Loopback-only, trust auth: this cluster exists to verify the application on
  # one machine. Trust is chosen over a password file so the verification run
  # has no credential to hold; a deployed cluster must not use it.
  "${BIN}/initdb" -D "${PGDATA}" -U postgres --auth-local=trust --auth-host=trust \
    -E UTF8 --locale=C >/dev/null
  {
    echo "# Local verification instance. Loopback only."
    echo "listen_addresses = '127.0.0.1'"
    echo "port = ${PGPORT}"
    echo "unix_socket_directories = '${PGRUN}'"
    echo "fsync = off"
    echo "synchronous_commit = off"
    echo "full_page_writes = off"
  } >>"${PGDATA}/postgresql.conf"
  echo "host all all 127.0.0.1/32 trust" >>"${PGDATA}/pg_hba.conf"
}

pg_running() {
  "${BIN}/pg_ctl" -D "${PGDATA}" status >/dev/null 2>&1
}

redis_running() {
  "${BIN}/redis-cli" -p "${REDISPORT}" ping >/dev/null 2>&1
}

start_postgres() {
  if pg_running; then
    echo "postgresql  already running on 127.0.0.1:${PGPORT}"
    return 0
  fi
  init_postgres
  "${BIN}/pg_ctl" -D "${PGDATA}" -l "${LOGS}/postgres.log" -w start >/dev/null

  # Create the application role and database if this is a fresh cluster.
  local exists
  exists="$("${BIN}/psql" -h 127.0.0.1 -p "${PGPORT}" -U postgres -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname='${PGUSER}'")"
  if [[ -z "${exists}" ]]; then
    "${BIN}/psql" -h 127.0.0.1 -p "${PGPORT}" -U postgres -v ON_ERROR_STOP=1 -q \
      -c "CREATE ROLE ${PGUSER} LOGIN PASSWORD '${PGPASSWORD}' SUPERUSER"
  fi

  exists="$("${BIN}/psql" -h 127.0.0.1 -p "${PGPORT}" -U postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='${PGDATABASE}'")"
  if [[ -z "${exists}" ]]; then
    "${BIN}/psql" -h 127.0.0.1 -p "${PGPORT}" -U postgres -v ON_ERROR_STOP=1 -q \
      -c "CREATE DATABASE ${PGDATABASE} OWNER ${PGUSER}"
  fi

  echo "postgresql  started on 127.0.0.1:${PGPORT} (database ${PGDATABASE})"
}

start_redis() {
  if redis_running; then
    echo "redis       already running on 127.0.0.1:${REDISPORT}"
    return 0
  fi
  mkdir -p "${REDISDATA}" "${LOGS}"
  "${BIN}/redis-server" \
    --port "${REDISPORT}" \
    --bind 127.0.0.1 \
    --dir "${REDISDATA}" \
    --logfile "${LOGS}/redis.log" \
    --daemonize yes \
    --maxmemory 256mb \
    --maxmemory-policy noeviction
  echo "redis       started on 127.0.0.1:${REDISPORT}"
}

stop_all() {
  if [[ -d "${PGDATA}/base" ]] && pg_running; then
    "${BIN}/pg_ctl" -D "${PGDATA}" -m fast -w stop >/dev/null
    echo "postgresql  stopped"
  fi
  if redis_running; then
    "${BIN}/redis-cli" -p "${REDISPORT}" shutdown nosave >/dev/null 2>&1 || true
    echo "redis       stopped"
  fi
}

status() {
  echo "runtime:    ${RUNTIME}"
  if [[ -d "${PGDATA}/base" ]] && pg_running; then
    echo "postgresql: running on 127.0.0.1:${PGPORT} ($("${BIN}/postgres" --version))"
  else
    echo "postgresql: stopped"
  fi
  if redis_running; then
    echo "redis:      running on 127.0.0.1:${REDISPORT}"
  else
    echo "redis:      stopped"
  fi
}

case "${1:-up}" in
  setup) setup ;;
  up) require_runtime; setup >/dev/null 2>&1 || true; start_postgres; start_redis ;;
  down) require_runtime; stop_all ;;
  status) require_runtime; status ;;
  reset) require_runtime; stop_all; rm -rf "${PGDATA}" "${REDISDATA}" "${PGRUN}"; start_postgres; start_redis ;;
  *) die "unknown command '${1:-}'. Use: setup | up | down | status | reset" ;;
esac
