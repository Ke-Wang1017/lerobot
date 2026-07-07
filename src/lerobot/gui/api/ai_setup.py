"""``/ai_setup`` — the AI / MCP credential-setup page.

The friction-killer for the dev workflow: a dev on the LAN opens
``http://lerobot.local:8000/ai_setup``, names their device, picks scopes,
and gets either a one-line copy-paste command (Claude Code / Codex CLI /
Gemini CLI / Claude Desktop JSON) or the URL / Bearer / Headers field
mapping for GUI tools (Codex IDE plugin, etc.). No terminal required
beyond that paste.

This route is intentionally separate from the GUI's API surface — it
renders HTML for humans, not JSON for the embedded agent. The token
storage layer (``lerobot.mcp.auth.TokenStore``) is shared with the MCP
daemon: this page writes; the daemon reads.

Auth posture: on a trusted LAN this page is open. No host-level auth
layer is introduced here (see the plan's "no host-level auth layer"
non-goal). A future enhancement could gate it; for now, anyone on the
LAN can issue themselves a token.
"""

from __future__ import annotations

import html
import logging
import os
from functools import lru_cache

from fastapi import APIRouter, Depends, Form, HTTPException, Path, Request
from fastapi.responses import HTMLResponse, RedirectResponse

# MCP is an optional install path: `lerobot[gui]` is a valid base set; the
# `mcp` package only lands when you install `lerobot[mcp]`. If it isn't
# present, the rest of the GUI keeps booting and `/ai_setup` itself returns
# 503 with a clear message. Without this guard, missing `mcp` would crash
# the whole GUI server at import time.
try:
    from lerobot.mcp.auth import ALL_SCOPES, SCOPE_COMMENT, SCOPE_READ, TokenStore, default_token_store_path

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    ALL_SCOPES: tuple[str, ...] = ()
    SCOPE_COMMENT = "comment"
    SCOPE_READ = "read"
    TokenStore = None  # type: ignore[assignment,misc]
    default_token_store_path = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai_setup", tags=["ai_setup"])


# ── Configuration ──────────────────────────────────────────────────────────


# Token-store path resolution lives in lerobot.mcp.auth.default_token_store_path
# so the page, the mounted server (server.py:_mount_mcp), and the standalone
# CLI all agree — including on LEROBOT_MCP_TOKEN_STORE overrides.


def _resolve_mcp_url(request: Request) -> str:
    """MCP endpoint URL clients should connect to.

    Preference order:
      1. ``LEROBOT_MCP_PUBLIC_URL`` env var (admin override for tunneled / NAT setups).
      2. Derived from the incoming request's ``Host`` header → ``http://<host>/mcp``.
         This is the right answer ~always: the user reached this page via
         exactly the URL their client also needs to use.
    """
    override = os.environ.get("LEROBOT_MCP_PUBLIC_URL")
    if override:
        return override
    host = request.headers.get("host", "lerobot.local:8000")
    scheme = request.url.scheme  # picks up x-forwarded-proto via uvicorn proxy-headers
    return f"{scheme}://{host}/mcp"


def _require_mcp() -> None:
    """Raise 503 if `mcp` isn't installed.

    The `/ai_setup` page exists only to issue MCP bearer tokens; without
    the `mcp` package there's nothing to manage. We degrade the page
    gracefully (503 + clear message) instead of crashing the whole GUI
    at import time.
    """
    if not _MCP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "MCP support is not installed. Re-install with the `mcp` extra "
                "to enable the `/ai_setup` page: `pip install 'lerobot[mcp]'`."
            ),
        )


@lru_cache(maxsize=1)
def _store() -> TokenStore:
    _require_mcp()
    return TokenStore(default_token_store_path())


def get_token_store() -> TokenStore:
    """FastAPI dependency. Tests override this with ``app.dependency_overrides``."""
    return _store()


# ── HTML rendering ─────────────────────────────────────────────────────────


def _snippets(token: str, mcp_url: str) -> list[tuple[str, str, str]]:
    """One-line config snippets per supported MCP client.

    Returns list of (tab_id, display_name, snippet) triples. ``tab_id`` is
    used by the in-page tab JS to switch which snippet is shown.
    """
    e = html.escape
    # Flag syntax differs per client (verified against each tool's docs):
    #   - Claude Code / Gemini: URL is positional, bearer via --header.
    #   - Codex: --url flag, and the bearer must come from an env var (no inline
    #     token flag), so we emit an `export` line first.
    #   - Claude Desktop has no native remote-HTTP entry — it needs the
    #     `mcp-remote` stdio proxy.
    return [
        (
            "claude-cli",
            "Claude Code",
            f"claude mcp add --transport http lerobot {e(mcp_url)} \\\n  --header 'Authorization: Bearer {e(token)}'",
        ),
        (
            "codex-cli",
            "Codex CLI",
            f"export LEROBOT_MCP_TOKEN={e(token)}\n"
            f"codex mcp add lerobot \\\n  --url {e(mcp_url)} \\\n  --bearer-token-env-var LEROBOT_MCP_TOKEN",
        ),
        (
            "gemini-cli",
            "Gemini CLI",
            f"gemini mcp add --transport http lerobot {e(mcp_url)} \\\n  --header 'Authorization: Bearer {e(token)}'",
        ),
        (
            "claude-desktop",
            "Claude Desktop (JSON)",
            "// add to ~/.config/Claude/claude_desktop_config.json\n"
            "// Claude Desktop has no native HTTP server entry — proxy via mcp-remote\n"
            '"mcpServers": {\n'
            '  "lerobot": {\n'
            '    "command": "npx",\n'
            '    "args": [\n'
            '      "mcp-remote",\n'
            f'      "{e(mcp_url)}",\n'
            f'      "--header", "Authorization: Bearer {e(token)}"\n'
            "    ]\n"
            "  }\n"
            "}",
        ),
    ]


_PAGE_CSS = """
<style>
  :root {
    --bg:        #1a1a2e;
    --panel:    #1e2a4a;
    --panel-2: #16213e;
    --border:   #2a4a6f;
    --text:     #eee;
    --text-2:  #ccc;
    --muted:    #888;
    --accent:   #4fc3f7;
    --warn-bg:  rgba(255,193,7,0.08);
    --warn-fg:  #ffc107;
    --ok-bg:    rgba(79,195,247,0.08);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; min-height: 100vh; }
  body { padding: 24px 16px 64px; }
  .wrap { max-width: 880px; margin: 0 auto; }

  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; }
  header h1 { font-size: 22px; font-weight: 600; letter-spacing: 0.3px; }
  header .accent { color: var(--accent); }
  header .home-link { margin-left: auto; font-size: 12px; color: var(--muted); text-decoration: none; border: 1px solid var(--border); border-radius: 4px; padding: 4px 10px; }
  header .home-link:hover { color: var(--text-2); border-color: var(--accent); }
  .lede { color: var(--text-2); font-size: 13px; line-height: 1.5; margin-bottom: 24px; }

  section { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; margin: 16px 0; }
  section h2 { font-size: 14px; font-weight: 600; color: var(--accent); letter-spacing: 0.4px; text-transform: uppercase; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  section h2 .step { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; background: var(--accent); color: var(--panel-2); border-radius: 50%; font-size: 11px; font-weight: 700; }
  section h3 { font-size: 13px; font-weight: 600; color: var(--text); margin: 16px 0 8px; }

  /* Issue form */
  .form-row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
  .form-row label { font-size: 12px; color: var(--text-2); display: inline-flex; align-items: center; gap: 6px; }
  .form-row input[type=text] { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; font-family: inherit; font-size: 13px; min-width: 200px; }
  .form-row input[type=text]:focus { outline: none; border-color: var(--accent); }
  .scope-group { display: inline-flex; gap: 12px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 4px; background: var(--panel-2); }
  .scope-group label { font-family: monospace; font-size: 12px; }
  .scope-group input[type=checkbox] { accent-color: var(--accent); }

  button.primary, .primary { background: var(--accent); color: var(--panel-2); border: 1px solid var(--accent); border-radius: 4px; padding: 6px 14px; font-family: inherit; font-size: 12px; font-weight: 600; cursor: pointer; }
  button.primary:hover { filter: brightness(1.1); }
  button.ghost { background: transparent; color: var(--text-2); border: 1px solid var(--border); border-radius: 4px; padding: 4px 10px; font-family: inherit; font-size: 11px; cursor: pointer; }
  button.ghost:hover { color: var(--text); border-color: var(--accent); background: var(--ok-bg); }

  /* Token list */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }
  td { color: var(--text-2); }
  td.name { font-family: monospace; color: var(--text); }
  .scope-tag { display: inline-block; padding: 1px 6px; margin-right: 4px; border-radius: 3px; background: var(--ok-bg); color: var(--accent); font-size: 11px; font-family: monospace; }
  .empty { color: var(--muted); font-style: italic; font-size: 12px; }

  /* New-token panel */
  .new-token { background: var(--warn-bg); border: 1px solid var(--warn-fg); }
  .new-token h2 { color: var(--warn-fg); }
  .warn-line { color: var(--warn-fg); font-size: 12px; margin: 8px 0; }
  .bearer-row { display: flex; gap: 8px; align-items: stretch; margin-top: 8px; }
  .bearer-row pre { flex: 1; }

  pre { background: var(--panel-2); border: 1px solid var(--border); border-radius: 4px; padding: 10px 12px; font-family: 'SF Mono', Menlo, monospace; font-size: 12px; line-height: 1.5; color: var(--text); overflow-x: auto; white-space: pre-wrap; word-break: break-all; }

  /* Tabs */
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid var(--border); margin-top: 12px; }
  .tabs button { background: transparent; color: var(--muted); border: 1px solid transparent; border-bottom: none; border-radius: 4px 4px 0 0; padding: 6px 12px; font-family: inherit; font-size: 12px; cursor: pointer; }
  .tabs button:hover { color: var(--text-2); }
  .tabs button.active { color: var(--accent); border-color: var(--border); border-bottom-color: var(--panel); background: var(--panel); position: relative; top: 1px; }
  .tab-body { display: none; margin-top: 8px; }
  .tab-body.active { display: block; }
  .tab-body .row { display: flex; gap: 8px; align-items: stretch; }
  .tab-body .row pre { flex: 1; }

  /* Field mapping for GUI tools */
  dl.fields { display: grid; grid-template-columns: 100px 1fr auto; gap: 8px 12px; align-items: center; font-size: 12px; }
  dl.fields dt { color: var(--muted); font-family: monospace; }
  dl.fields dd { font-family: monospace; color: var(--text); background: var(--panel-2); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; }
  dl.fields dd.placeholder { color: var(--muted); font-style: italic; }

  .back { display: inline-block; margin-top: 12px; font-size: 12px; color: var(--muted); text-decoration: none; }
  .back:hover { color: var(--accent); }

  /* Inline warning block — amber-on-dark, sized to slot inside a section. */
  .warn-block { background: var(--warn-bg); border: 1px solid var(--warn-fg); border-left-width: 3px; border-radius: 4px; padding: 10px 12px; margin: 12px 0; font-size: 12px; color: var(--text-2); line-height: 1.5; }
  .warn-block strong { color: var(--warn-fg); display: block; margin-bottom: 4px; font-size: 12px; letter-spacing: 0.3px; text-transform: uppercase; }
  .warn-block code { color: var(--accent); font-family: monospace; }

  /* Native <details> with the lerobot palette. */
  details summary { cursor: pointer; list-style: none; user-select: none; }
  details summary::-webkit-details-marker { display: none; }
  details summary h2 { margin-bottom: 0; display: inline-flex; align-items: center; gap: 8px; }
  details summary h2::after { content: "▸"; color: var(--muted); font-size: 11px; margin-left: 4px; transition: transform 0.15s; }
  details[open] summary h2::after { transform: rotate(90deg); display: inline-block; }
</style>
"""


_TABS_JS = """
<script>
function copyText(targetSel, btn) {
  const el = document.querySelector(targetSel);
  if (!el) return;
  const text = el.dataset.copy ?? el.innerText;
  const flash = (ok) => {
    const orig = btn.innerText;
    btn.innerText = ok ? "Copied" : "Copy failed — select & ⌘/Ctrl-C";
    setTimeout(() => { btn.innerText = orig; }, ok ? 1200 : 2500);
  };
  // navigator.clipboard exists only in a secure context (HTTPS or localhost).
  // Over plain HTTP to a LAN host (http://lerobot.local:PORT) it's undefined,
  // so fall back to a hidden-textarea + execCommand, which works there.
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => flash(true), () => flash(_legacyCopy(text)));
    return;
  }
  flash(_legacyCopy(text));
}
function _legacyCopy(text) {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
function switchSnippet(id) {
  document.querySelectorAll('.tabs button').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === id);
  });
  document.querySelectorAll('.tab-body').forEach(b => {
    b.classList.toggle('active', b.id === 'tab-' + id);
  });
}
</script>
"""


def _render_listing(store: TokenStore) -> str:
    rows = store.list_tokens(include_revoked=False)
    if not rows:
        return '<p class="empty">No active tokens yet. Issue one below.</p>'
    body = "<table><thead><tr><th>Name</th><th>Scopes</th><th>Created</th><th>Last used</th><th></th></tr></thead><tbody>"
    for r in rows:
        scopes_html = "".join(f'<span class="scope-tag">{html.escape(s)}</span>' for s in r["scopes"])
        last = r["last_used_at"] or "—"
        body += (
            "<tr>"
            f'<td class="name">{html.escape(r["name"])}</td>'
            f"<td>{scopes_html}</td>"
            f"<td>{html.escape(r['created_at'])}</td>"
            f"<td>{html.escape(last)}</td>"
            f'<td style="text-align:right;"><form method="post" action="/ai_setup/tokens/{html.escape(r["name"])}/revoke" style="display:inline">'
            f'<button type="submit" class="ghost" onclick="return confirm(\'Revoke {html.escape(r["name"])}?\')">Revoke</button>'
            "</form></td>"
            "</tr>"
        )
    body += "</tbody></table>"
    return body


def _render_new_form(default_scopes: tuple[str, ...] = (SCOPE_READ, SCOPE_COMMENT)) -> str:
    # Scope checkboxes. read + comment are checked by default; edit and operate
    # are opt-in. operate is the highest tier — it grants hardware + training
    # control (incl. starting cloud runs, which bill), so it's selectable but
    # never default. (It was disabled until the operate-tier tools existed; the
    # training_* tools are the first, so it's now grantable.)
    scope_tooltips = {
        SCOPE_READ: "List/read datasets, models, robots, runs. No state changes.",
        SCOPE_COMMENT: "Plus write sidecar comments — notes that persist across AI sessions, never touch canonical data.",
        "edit": "Plus mutate canonical state: dataset edits (delete/trim/feature-set), Hub uploads, profile config.",
        "operate": "Plus run hardware + training: teleop, record, replay, start/stop training runs (cloud starts bill), recovery.",
    }
    boxes = []
    for s in ALL_SCOPES:
        checked = "checked" if s in default_scopes else ""
        tooltip = scope_tooltips.get(s, "")
        title = f' title="{tooltip}"' if tooltip else ""
        boxes.append(f'<label{title}><input type="checkbox" name="scope" value="{s}" {checked}> {s}</label>')
    return (
        '<form method="post" action="/ai_setup/tokens">'
        '<div class="form-row">'
        '<label>Device name <input type="text" name="name" required '
        'placeholder="e.g. alice-laptop" pattern="[A-Za-z0-9._-]+" autofocus></label>'
        f'<div class="scope-group">{"".join(boxes)}</div>'
        '<button type="submit" class="primary">Issue token</button>'
        "</div>"
        "</form>"
    )


_TOKEN_PLACEHOLDER = "<YOUR_TOKEN>"  # noqa: S105 — literal placeholder, not a secret  # nosec B105


def _render_install_snippets(token: str, mcp_url: str, *, id_prefix: str) -> str:
    """Per-tool snippet tabs + the URL/Bearer/Headers field-mapping table.

    Used in two places:
      - the post-issue panel, with the actual ``token`` filled in;
      - the always-available installation reference on the index page,
        with ``_TOKEN_PLACEHOLDER`` so it's useful even without a fresh
        token issue.

    ``id_prefix`` namespaces DOM ids so the index page (collapsed reference)
    and the post-issue panel can coexist on the same page if we ever want
    to render both, and so per-tab JS targets the right ``<pre>``.
    """
    e = html.escape
    snippets = _snippets(token, mcp_url)

    tab_buttons = "".join(
        f'<button type="button" data-tab="{id_prefix}-{tid}" '
        f"onclick=\"switchSnippet('{id_prefix}-{tid}')\" "
        f'class="{"active" if i == 0 else ""}">{e(label)}</button>'
        for i, (tid, label, _) in enumerate(snippets)
    )
    tab_bodies = "".join(
        f'<div class="tab-body {"active" if i == 0 else ""}" id="tab-{id_prefix}-{tid}">'
        f'<div class="row"><pre data-copy="{e(s, quote=True)}">{s}</pre>'
        f'<button type="button" class="ghost" onclick="copyText(\'#tab-{id_prefix}-{tid} pre\', this)">Copy</button>'
        "</div></div>"
        for i, (tid, _, s) in enumerate(snippets)
    )

    return (
        "<h3>Connect your tool</h3>"
        f'<div class="tabs">{tab_buttons}</div>'
        f"{tab_bodies}"
        '<div class="warn-block">'
        "<strong>⚠ Some clients can't resolve <code>.local</code></strong>"
        "Codex and other Go-based MCP clients use a pure-Go DNS resolver that "
        "bypasses mDNS / Bonjour. If your tool hangs connecting to "
        "<code>lerobot.local</code>, swap the host for the LeRobot machine's "
        "LAN IP (find it on the host with <code>ip addr</code>, or in your "
        "router's client list). Claude Code, Codex CLI, Gemini CLI, Claude "
        "Desktop, <code>curl</code> and <code>ping</code> all go through the "
        "system resolver and work fine with <code>.local</code>."
        "</div>"
        '<div class="note" style="margin-top:8px;">'
        "<strong>If the server moves</strong> (different port or host), this token stays "
        "valid — just change the <em>URL</em> in your AI tool; no need to re-issue. For a "
        "stable URL, run the GUI on a fixed port, or define <code>LEROBOT_MCP_PUBLIC_URL</code> "
        "to pin a canonical address."
        "</div>"
        "<h3>Tool asks for fields separately? Use these</h3>"
        '<dl class="fields">'
        "<dt>URL</dt>"
        f'<dd id="{id_prefix}-field-url" data-copy="{e(mcp_url, quote=True)}">{e(mcp_url)}</dd>'
        f'<dd><button type="button" class="ghost" onclick="copyText(\'#{id_prefix}-field-url\', this)">Copy</button></dd>'
        "<dt>Bearer</dt>"
        f'<dd id="{id_prefix}-field-bearer" data-copy="{e(token, quote=True)}">{e(token)}</dd>'
        f'<dd><button type="button" class="ghost" onclick="copyText(\'#{id_prefix}-field-bearer\', this)">Copy</button></dd>'
        "<dt>Headers</dt>"
        '<dd class="placeholder">(leave empty — the Bearer field above is enough)</dd>'
        "<dd></dd>"
        "</dl>"
    )


def _render_install_reference(mcp_url: str) -> str:
    """Always-available collapsed reference of install commands.

    The ``<details>`` element starts closed so the index page stays focused
    on Issue + Active tokens. Expanding reveals the same snippets the
    post-issue panel shows, but with ``<YOUR_TOKEN>`` as the placeholder —
    so a user who's already saved their bearer doesn't need to re-issue
    one just to see the install command.
    """
    snippets_html = _render_install_snippets(token=_TOKEN_PLACEHOLDER, mcp_url=mcp_url, id_prefix="ref")
    return (
        "<details>"
        '<summary><h2><span class="step">3</span>Installation snippets</h2></summary>'
        '<p class="lede" style="margin-top:8px;">Reference only — replace '
        f'<code style="color:var(--accent);font-family:monospace;">{_TOKEN_PLACEHOLDER}</code> '
        "with the bearer you saved when issuing the token. "
        "Lost your token? Just issue a new one above and the panel will "
        "show the bearer filled in.</p>"
        f"{snippets_html}"
        "</details>"
    )


def _render_new_token_panel(name: str, token: str, scopes: list[str], mcp_url: str) -> str:
    e = html.escape
    return (
        '<section class="new-token">'
        f'<h2>Token created for <span style="font-family:monospace;color:var(--text);">{e(name)}</span></h2>'
        f'<p style="font-size:12px;color:var(--text-2);">Scopes: {"".join(f"<span class=scope-tag>{e(s)}</span>" for s in scopes)}</p>'
        '<p class="warn-line">⚠ The bearer below is shown only once. Save it now — it cannot be recovered.</p>'
        '<div class="bearer-row">'
        f'<pre id="bearer" data-copy="{e(token, quote=True)}">{e(token)}</pre>'
        '<button type="button" class="ghost" onclick="copyText(\'#bearer\', this)">Copy</button>'
        "</div>"
        f"{_render_install_snippets(token=token, mcp_url=mcp_url, id_prefix='new')}"
        '<a class="back" href="/ai_setup">← Back to token list</a>'
        "</section>"
    )


def _page(body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>LeRobot — AI setup</title>"
        f"{_PAGE_CSS}{_TABS_JS}"
        "</head><body>"
        '<div class="wrap">'
        "<header>"
        '<h1>LeRobot — <span class="accent">AI setup</span></h1>'
        '<a class="home-link" href="/">← Back to GUI</a>'
        "</header>"
        '<p class="lede">Issue a bearer token, paste it into your AI tool, and the tool can read datasets / tag episodes / drive your open GUI tab through MCP. Anyone on the LAN can issue themselves a token — revoke any time below.</p>'
        f"{body}"
        "</div></body></html>"
    )


# ── Routes ─────────────────────────────────────────────────────────────────


@router.get("", response_class=HTMLResponse)
def index(request: Request, store: TokenStore = Depends(get_token_store)) -> HTMLResponse:
    """Token list + new-token form + collapsed installation reference."""
    mcp_url = _resolve_mcp_url(request)
    body = (
        '<section><h2><span class="step">1</span>Issue a token</h2>'
        f"{_render_new_form()}</section>"
        '<section><h2><span class="step">2</span>Active tokens</h2>'
        f"{_render_listing(store)}</section>"
        f"<section>{_render_install_reference(mcp_url)}</section>"
    )
    return HTMLResponse(_page(body))


@router.post("/tokens", response_class=HTMLResponse)
def create_token(
    request: Request,
    name: str = Form(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$"),
    scope: list[str] = Form(...),
    store: TokenStore = Depends(get_token_store),
) -> HTMLResponse:
    """Issue a new token and render a one-time view with copy-paste snippets."""
    invalid = [s for s in scope if s not in ALL_SCOPES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid scope(s): {invalid}")
    try:
        token = store.issue(name=name, scopes=scope)
    except ValueError as e:
        # Most likely: name already exists.
        raise HTTPException(status_code=409, detail=str(e)) from e
    body = _render_new_token_panel(
        name=name, token=token, scopes=sorted(set(scope)), mcp_url=_resolve_mcp_url(request)
    )
    return HTMLResponse(_page(body))


@router.post("/tokens/{name}/revoke")
def revoke_token(
    # Same alphabet as create_token so the URL surface matches the create-side
    # contract; arbitrary names would just be a 404 probe oracle.
    name: str = Path(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$"),
    store: TokenStore = Depends(get_token_store),
) -> RedirectResponse:
    if not store.revoke(name):
        raise HTTPException(status_code=404, detail=f"No active token named {name!r}")
    # 303 turns the POST into a GET on the redirect so refreshing isn't a re-POST.
    return RedirectResponse(url="/ai_setup", status_code=303)
