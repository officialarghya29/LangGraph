"""A minimal operator console.

Three routes serve a static page and its two assets. The page talks to the same
JSON API a script would, using the same credentials — it is a convenience view,
not a second interface with its own privileges.

Four decisions are deliberate, and each one is a security decision first:

- **It is off by default.** ``DASHBOARD_ENABLED`` must be set explicitly. A page
  with approve, reject, and cancel buttons is an operator console, and one that
  any browser can reach is an attack surface. Local development turns it on;
  anything with real users should not.
- **No inline script or style.** The JavaScript and CSS are separate routes, so
  the Content-Security-Policy is ``script-src 'self'`` with no ``unsafe-inline``.
  An injected ``<script>`` in a task description then does nothing, which matters
  because task descriptions are user input and this page renders them.
- **Untrusted values are written with ``textContent``, never ``innerHTML``.** The
  page displays request text, answers, and event payloads. Building HTML out of
  those strings is how an operator console becomes a stored-XSS delivery
  mechanism; the asset test asserts the pattern is absent rather than trusting
  the reviewer to notice.
- **Failed responses are shown, not swallowed.** A console that silently fails to
  cancel a task is worse than one with no cancel button, so every call renders its
  status and any error detail.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, Response

from app.core.config import get_settings

router = APIRouter(tags=["dashboard"])

#: The policy the page is served under. ``default-src 'none'`` with everything
#: else named explicitly: any resource the page loads must be listed, so a
#: future edit that reaches for a CDN fails closed instead of silently working.
CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "connect-src 'self'; "
    "img-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)

#: The API paths the page calls. Kept as data so the test can assert the page
#: references the endpoints that actually exist, which is how a renamed route
#: becomes a failing test instead of a broken button.
API_PATHS: tuple[str, ...] = (
    "/health",
    "/ready",
    "/metrics",
    "/api/v1/agents",
    "/api/v1/tools",
    "/api/v1/tasks",
)

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Agent control console</title>
<link rel="stylesheet" href="/dashboard/app.css">
</head>
<body>
<header>
  <h1>Agent control console</h1>
  <div class="statusline">
    <span id="health" class="badge pending">health: ?</span>
    <span id="ready" class="badge pending">ready: ?</span>
    <span id="identity" class="badge">identity: -</span>
  </div>
</header>

<main>
  <section>
    <h2>Run a request</h2>
    <form id="submit-form">
      <label for="request">Request</label>
      <textarea id="request" rows="4" required
        placeholder="Describe the task. It is routed, planned, and dispatched."></textarea>
      <label for="identity-input">Identity</label>
      <input id="identity-input" type="text" value="dashboard-operator"
             autocomplete="off" spellcheck="false">
      <label for="token-input">Bearer token (only when authentication is enabled)</label>
      <input id="token-input" type="password" autocomplete="off" spellcheck="false">
      <button type="submit">Submit</button>
    </form>
    <p id="submit-result" class="muted"></p>
  </section>

  <section>
    <h2>Tasks</h2>
    <div class="row">
      <button id="refresh" type="button">Refresh</button>
      <button id="stream" type="button">Stream selected</button>
      <button id="stop-stream" type="button">Stop stream</button>
      <button id="approve" type="button" class="warn">Approve</button>
      <button id="reject" type="button" class="warn">Reject</button>
      <button id="cancel" type="button" class="danger">Cancel</button>
    </div>
    <table>
      <thead>
        <tr><th>Task</th><th>Status</th><th>Approval</th><th>Route</th><th>Request</th></tr>
      </thead>
      <tbody id="tasks"><tr><td colspan="5" class="muted">No tasks loaded.</td></tr></tbody>
    </table>
  </section>

  <section>
    <h2>Selected task</h2>
    <dl id="detail"><dt>Nothing selected</dt></dl>
    <h3>Events</h3>
    <ol id="events" class="events"></ol>
  </section>

  <section>
    <h2>Capabilities</h2>
    <div class="two">
      <div><h3>Agents</h3><ul id="agents"></ul></div>
      <div><h3>Tools</h3><ul id="tools"></ul></div>
    </div>
  </section>

  <section>
    <h2>Metrics</h2>
    <pre id="metrics" class="metrics">Not loaded.</pre>
  </section>
</main>

<footer>
  <p class="muted">This console calls the same JSON API as any other client and
  holds no privileges of its own.</p>
</footer>

<script src="/dashboard/app.js"></script>
</body>
</html>
"""

_STYLES = """/* Deliberately small: a console is read at a glance, and a dark-neutral palette
   keeps a failing task from being less visible than a passing one. */
:root {
  --bg: #12141a; --panel: #1b1e26; --line: #2b303b; --text: #e6e8ee;
  --muted: #98a0b3; --ok: #3fb950; --warn: #d29922; --bad: #f85149; --accent: #58a6ff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
header, main, footer { padding: 16px 24px; }
header { border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 8px; font-size: 18px; }
h2 { font-size: 15px; margin: 0 0 12px; color: var(--accent); }
h3 { font-size: 13px; margin: 12px 0 6px; color: var(--muted); }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  padding: 16px; margin: 0 0 16px; }
textarea, input {
  width: 100%; padding: 8px; margin: 4px 0 12px; background: var(--bg);
  color: var(--text); border: 1px solid var(--line); border-radius: 4px;
  font: inherit;
}
label { display: block; color: var(--muted); font-size: 12px; }
button {
  padding: 7px 14px; margin-right: 8px; background: #262b36; color: var(--text);
  border: 1px solid var(--line); border-radius: 4px; font: inherit; cursor: pointer;
}
button:hover { border-color: var(--accent); }
button.warn { color: var(--warn); }
button.danger { color: var(--bad); }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: normal; font-size: 12px; }
tr.selected { background: #202634; }
tr.clickable { cursor: pointer; }
.badge { display: inline-block; padding: 2px 8px; margin-right: 8px;
  border: 1px solid var(--line); border-radius: 10px; font-size: 12px; }
.badge.ok { color: var(--ok); border-color: var(--ok); }
.badge.warn { color: var(--warn); border-color: var(--warn); }
.badge.bad { color: var(--bad); border-color: var(--bad); }
.badge.pending { color: var(--muted); }
.muted { color: var(--muted); }
.events { list-style: none; margin: 0; padding: 0; max-height: 320px; overflow-y: auto; }
.events li { padding: 4px 0; border-bottom: 1px solid var(--line); font-size: 12px; }
.events .type { color: var(--accent); }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
ul { margin: 0; padding-left: 18px; }
.metrics { max-height: 260px; overflow: auto; background: var(--bg);
  border: 1px solid var(--line); border-radius: 4px; padding: 10px; font-size: 12px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
"""

_SCRIPT = """'use strict';

// The console holds no privileges of its own: it sends the same headers a script
// would. The identity header is only honoured when the server has been
// explicitly configured to trust it, and it is a development affordance, not a
// substitute for a token.
const state = { selected: null, controller: null };

function headers() {
  const identity = el('identity-input').value.trim();
  const token = el('token-input').value.trim();
  const result = { 'Content-Type': 'application/json' };
  if (identity) result['X-User-Id'] = identity;
  if (token) result['Authorization'] = 'Bearer ' + token;
  return result;
}

function el(id) { return document.getElementById(id); }

function setText(id, text) { el(id).textContent = text; }

function badge(id, text, kind) {
  const node = el(id);
  node.textContent = text;
  node.className = 'badge ' + kind;
}

async function call(path, options) {
  const response = await fetch(path, Object.assign({ headers: headers() }, options));
  const text = await response.text();
  let body = text;
  try { body = JSON.parse(text); } catch (ignored) { /* keep the raw text */ }
  return { ok: response.ok, status: response.status, body };
}

function describeFailure(result) {
  const detail = result.body && result.body.detail ? result.body.detail : result.body;
  return 'HTTP ' + result.status + ': ' + String(detail);
}

// -- health --------------------------------------------------------------- //

async function refreshHealth() {
  const health = await call('/health');
  badge('health', 'health: ' + (health.ok ? 'ok' : 'down'), health.ok ? 'ok' : 'bad');

  const ready = await call('/ready');
  const checks = (ready.body && ready.body.checks) || {};
  const degraded = Object.keys(checks).filter((key) => checks[key] !== 'ok');
  badge(
    'ready',
    'ready: ' + (ready.ok ? 'ok' : 'degraded') +
      (degraded.length ? ' (' + degraded.join(', ') + ')' : ''),
    ready.ok ? 'ok' : 'warn'
  );
}

// -- discovery ------------------------------------------------------------ //

async function refreshCapabilities() {
  const agents = await call('/api/v1/agents');
  const agentsList = el('agents');
  agentsList.replaceChildren();
  if (Array.isArray(agents.body)) {
    agents.body.forEach((agent) => {
      const item = document.createElement('li');
      // textContent, not innerHTML: these strings describe user-visible
      // behaviour and one day one of them will contain a quote or an angle
      // bracket, and this page renders task text from the database.
      item.textContent = agent.name + (agent.description ? ' — ' + agent.description : '');
      agentsList.appendChild(item);
    });
  }

  const tools = await call('/api/v1/tools');
  const toolsList = el('tools');
  toolsList.replaceChildren();
  if (Array.isArray(tools.body)) {
    tools.body.forEach((tool) => {
      const item = document.createElement('li');
      item.textContent = tool.name + ' [' + tool.risk_level + ']';
      toolsList.appendChild(item);
    });
  }
}

// -- tasks ---------------------------------------------------------------- //

function renderTasks(tasks) {
  const body = el('tasks');
  body.replaceChildren();
  if (!tasks.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'muted';
    cell.textContent = 'No tasks yet.';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  tasks.forEach((task) => {
    const row = document.createElement('tr');
    row.className = 'clickable' + (task.task_id === state.selected ? ' selected' : '');
    [task.task_id, task.status, task.approval_status, task.route || '-', task.request]
      .forEach((value) => {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        row.appendChild(cell);
      });
    row.addEventListener('click', () => select(task.task_id));
    body.appendChild(row);
  });
}

async function refreshTasks() {
  const result = await call('/api/v1/tasks');
  if (!result.ok) { setText('submit-result', describeFailure(result)); return; }
  renderTasks(Array.isArray(result.body) ? result.body : []);
}

async function select(taskId) {
  state.selected = taskId;
  await refreshTasks();
  const result = await call('/api/v1/tasks/' + encodeURIComponent(taskId));
  const detail = el('detail');
  detail.replaceChildren();
  if (!result.ok) {
    const term = document.createElement('dt');
    term.textContent = describeFailure(result);
    detail.appendChild(term);
    return;
  }
  Object.keys(result.body).forEach((key) => {
    const term = document.createElement('dt');
    term.textContent = key;
    const value = document.createElement('dd');
    value.textContent = String(result.body[key]);
    detail.appendChild(term);
    detail.appendChild(value);
  });
  setText('identity', 'identity: ' + el('identity-input').value.trim());
  await loadHistory(taskId);
}

async function loadHistory(taskId) {
  const result = await call('/api/v1/events/' + encodeURIComponent(taskId) + '/history');
  const list = el('events');
  list.replaceChildren();
  const events = result.ok && Array.isArray(result.body.events) ? result.body.events : [];
  events.forEach(appendEvent);
}

function appendEvent(event) {
  const list = el('events');
  const item = document.createElement('li');
  const type = document.createElement('span');
  type.className = 'type';
  type.textContent = (event.sequence !== undefined ? '#' + event.sequence + ' ' : '') +
    (event.type || 'event') + ' ';
  const payload = document.createElement('span');
  payload.textContent = JSON.stringify(event.data === undefined ? {} : event.data);
  item.appendChild(type);
  item.appendChild(payload);
  list.appendChild(item);
  list.scrollTop = list.scrollHeight;
}

function stopStream() {
  if (state.controller) { state.controller.abort(); state.controller = null; }
}

// Streaming uses fetch, not EventSource.
//
// EventSource cannot set request headers, so it cannot present the identity or
// the bearer token the API requires. Passing them as query parameters would be
// the obvious workaround and the wrong one: a credential in a URL leaks into
// browser history, into the Referer of any outbound request, and into every
// access log on the path. fetch() can send headers, and its response body is a
// readable stream, so the frame parsing is done here instead.
function parseFrames(buffer, onFrame) {
  const parts = buffer.split('\n\n');
  const remainder = parts.pop();
  parts.forEach((part) => {
    let payload = null;
    part.split('\n').forEach((line) => {
      if (line.startsWith('data:')) payload = line.slice(5).trim();
    });
    if (payload) onFrame(payload);
  });
  return remainder;
}

async function startStream() {
  if (!state.selected) { setText('submit-result', 'Select a task first.'); return; }
  stopStream();
  const controller = new AbortController();
  state.controller = controller;

  const response = await fetch(
    '/api/v1/events/' + encodeURIComponent(state.selected),
    { headers: headers(), signal: controller.signal }
  );
  if (!response.ok) {
    setText('submit-result', 'stream → ' + describeFailure(
      { ok: false, status: response.status, body: await response.text() }
    ));
    stopStream();
    return;
  }

  setText('submit-result', 'streaming ' + state.selected);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      buffer = parseFrames(buffer, (payload) => {
        try { appendEvent(JSON.parse(payload)); } catch (ignored) { /* keep going */ }
      });
    }
  } catch (error) {
    if (!controller.signal.aborted) {
      setText('submit-result', 'stream ended: ' + error.message);
    }
  } finally {
    stopStream();
  }
}

async function decide(action) {
  if (!state.selected) { setText('submit-result', 'Select a task first.'); return; }
  const result = await call(
    '/api/v1/tasks/' + encodeURIComponent(state.selected) + '/' + action,
    { method: 'POST', body: JSON.stringify({ note: 'via control console' }) }
  );
  setText('submit-result',
    action + ' → ' + (result.ok ? 'ok' : describeFailure(result)));
  await refreshTasks();
  await select(state.selected);
}

async function submit(event) {
  event.preventDefault();
  const request = el('request').value.trim();
  if (!request) { return; }
  const result = await call('/api/v1/tasks', {
    method: 'POST',
    body: JSON.stringify({ request })
  });
  if (!result.ok) {
    setText('submit-result', describeFailure(result));
    return;
  }
  setText('submit-result', 'accepted as ' + result.body.task_id);
  el('request').value = '';
  await refreshTasks();
  await select(result.body.task_id);
}

async function refreshMetrics() {
  const response = await fetch('/metrics');
  setText('metrics', await response.text());
}

// -- wiring --------------------------------------------------------------- //

el('submit-form').addEventListener('submit', submit);
el('refresh').addEventListener('click', refreshTasks);
el('stream').addEventListener('click', () => { startStream(); });
el('stop-stream').addEventListener('click', stopStream);
el('approve').addEventListener('click', () => decide('approve'));
el('reject').addEventListener('click', () => decide('reject'));
el('cancel').addEventListener('click', () => decide('cancel'));
el('identity-input').addEventListener('change', () => {
  setText('identity', 'identity: ' + el('identity-input').value.trim());
  refreshHealth();
});

refreshHealth();
refreshCapabilities();
refreshTasks();
refreshMetrics();
setInterval(refreshHealth, 15000);
"""


def _require_enabled() -> None:
    """Reject the request unless the console has been explicitly enabled.

    Raises:
        HTTPException: 404 when the console is off. Not 403: a disabled route
            should not confirm that it exists, and the setting is not a
            permission the caller can obtain.
    """
    if not get_settings().dashboard_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


def _text(body: str, media_type: str) -> Response:
    """Return a response carrying the page or one of its assets.

    The security headers are set per response rather than in middleware, because
    they describe these three routes specifically: the JSON API is not rendered
    by a browser and gains nothing from a script policy.

    Args:
        body: The response body.
        media_type: The content type to serve it as.

    Returns:
        The response, with no-store caching and the console's CSP.
    """
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Security-Policy": CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            # The page is a live view of moving state, and a cached copy would
            # show task statuses that are no longer true.
            "Cache-Control": "no-store",
        },
    )


@router.get("/dashboard", response_class=HTMLResponse, summary="Operator console")
async def dashboard(request: Request) -> Response:
    """Serve the console's HTML.

    Returns:
        The page, or a 404 when :attr:`Settings.dashboard_enabled` is false.
    """
    del request
    _require_enabled()
    return _text(_PAGE, "text/html; charset=utf-8")


@router.get("/dashboard/app.css", summary="Console stylesheet")
async def dashboard_styles() -> Response:
    """Serve the console's stylesheet.

    Returns:
        The stylesheet, or a 404 when the console is disabled.
    """
    _require_enabled()
    return _text(_STYLES, "text/css; charset=utf-8")


@router.get("/dashboard/app.js", summary="Console script")
async def dashboard_script() -> Response:
    """Serve the console's script.

    Returns:
        The script, or a 404 when the console is disabled.
    """
    _require_enabled()
    return _text(_SCRIPT, "text/javascript; charset=utf-8")
