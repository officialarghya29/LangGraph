"""Execution-event streaming.

A client can follow a task as it runs instead of polling it. The stream carries
only the fixed, client-safe event vocabulary defined in
:mod:`app.schemas.events`: what a node started and finished, which tool ran,
whether a human is needed. It never carries chain-of-thought, a prompt, a raw
model response, or a credential — those are not filtered out here, they are
never placed on the event bus at all.

Two decisions worth stating:

- **The stream tails the durable event log, not a cache.** Redis would be
  faster, but a cache is evictable and a stream that silently loses events is
  worse than one that is a few hundred milliseconds behind. The cache is used
  for the dashboard ticker, where eventual consistency is acceptable.
- **Reconnection is first-class.** Every event carries a sequence number and the
  endpoint accepts ``after_seq``, so a client that drops can resume from exactly
  where it stopped rather than replaying or missing a window.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.auth import PrincipalDep
from app.models.execution import TaskStatus

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["events"])

#: How often the durable log is polled for new events.
POLL_INTERVAL_SECONDS = 0.4
#: A comment is sent this often so an idle stream is not closed by a proxy.
HEARTBEAT_SECONDS = 15.0
#: Hard ceiling on one connection. A client reconnects with ``after_seq``.
MAX_STREAM_SECONDS = 900.0
#: Events returned per poll.
BATCH_SIZE = 200

#: Statuses after which no further event can arrive.
_TERMINAL = {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}


def _format_event(event: dict[str, object]) -> str:
    """Serialise one event as a server-sent event frame.

    The payload is JSON on a single line: a literal newline inside a ``data``
    field would terminate the frame early and corrupt the stream.
    """
    body = json.dumps(event, default=str, separators=(",", ":"))
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {body}\n\n"


async def _event_stream(
    request: Request,
    task_id: str,
    principal_user: str,
    after_seq: int,
) -> AsyncIterator[str]:
    """Yield server-sent event frames for a task until it finishes.

    Stops when the task reaches a terminal state *and* every event has been
    drained, so a client never exits the stream having missed the final event.
    """
    store = request.app.state.task_store
    last_seq = after_seq
    started = asyncio.get_running_loop().time()
    idle_for = 0.0

    yield _format_event(
        {
            "seq": 0,
            "type": "stream_open",
            "payload": {"task_id": task_id, "after_seq": after_seq},
        }
    )

    while True:
        if await request.is_disconnected():
            logger.info("stream.client_disconnected", extra={"task_id": task_id})
            return

        try:
            events = await store.list_events(task_id, after_seq=last_seq)
        except Exception as exc:
            logger.warning(
                "stream.read_failed",
                extra={"task_id": task_id, "error": type(exc).__name__},
            )
            events = []

        for event in events:
            last_seq = max(last_seq, int(str(event["seq"])))
            yield _format_event(event)
            idle_for = 0.0

        record = await store.get_for_user(task_id, principal_user)
        if record is None:
            # Ownership changed or the task was deleted mid-stream. Ending the
            # stream is the only safe move; continuing could leak later events.
            yield _format_event(
                {"seq": last_seq + 1, "type": "stream_closed", "payload": {"reason": "gone"}}
            )
            return

        if record.status.value in _TERMINAL and not events:
            yield _format_event(
                {
                    "seq": last_seq + 1,
                    "type": "stream_closed",
                    "payload": {"status": record.status.value},
                }
            )
            return

        elapsed = asyncio.get_running_loop().time() - started
        if elapsed >= MAX_STREAM_SECONDS:
            # Bounded rather than open-ended: a client that reconnects with
            # after_seq loses nothing, while a connection that lives forever
            # holds a worker and a database pool slot.
            yield _format_event(
                {"seq": last_seq + 1, "type": "stream_closed", "payload": {"reason": "timeout"}}
            )
            return

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        idle_for += POLL_INTERVAL_SECONDS
        if idle_for >= HEARTBEAT_SECONDS:
            idle_for = 0.0
            # A comment frame keeps the connection alive through an idle proxy.
            yield ": keep-alive\n\n"


@router.get(
    "/events/{task_id}",
    summary="Stream a task's execution events",
    response_class=StreamingResponse,
)
async def stream_task_events(
    task_id: str,
    request: Request,
    principal: PrincipalDep,
    after_seq: int = Query(default=0, ge=0, description="Resume after this sequence number"),
) -> StreamingResponse:
    """Follow a task's execution as server-sent events.

    Args:
        task_id: The task to follow.
        request: Incoming request, used for disconnect detection.
        principal: The caller, who must own the task.
        after_seq: Resume point, so a reconnect does not replay.

    Returns:
        A ``text/event-stream`` response.

    Raises:
        HTTPException: 404 when the task does not exist or belongs to somebody
            else. The two are deliberately indistinguishable.
    """
    store = request.app.state.task_store
    record = await store.get_for_user(task_id, principal.user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such task")

    return StreamingResponse(
        _event_stream(request, task_id, principal.user_id, after_seq),
        media_type="text/event-stream",
        headers={
            # Proxies buffer by default, which would defeat streaming entirely.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/events/{task_id}/history", summary="Read a task's recorded events")
async def task_event_history(
    task_id: str,
    request: Request,
    principal: PrincipalDep,
    after_seq: int = Query(default=0, ge=0),
) -> list[dict[str, object]]:
    """Return a task's recorded events without streaming.

    Useful for rendering a completed run, and for a client that would rather
    page than hold a connection open.

    Raises:
        HTTPException: 404 when the task is absent or not owned by the caller.
    """
    store = request.app.state.task_store
    if await store.get_for_user(task_id, principal.user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such task")
    return list(await store.list_events(task_id, after_seq=after_seq))
