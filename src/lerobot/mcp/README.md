# `lerobot.mcp` — AI-native LeRobot

Connect your AI tool of choice (Claude Code, Codex CLI, Cursor, Claude
Desktop, scripts) to LeRobot via the [Model Context
Protocol](https://modelcontextprotocol.io). The AI browses your
datasets, leaves comments that survive across sessions and AI tools,
and drives the GUI tab you have open. Your existing AI subscription
powers it — no separate API key, no LeRobot AI account.

**Jump to:**

- [For operators — connecting your AI tool](#for-operators--connecting-your-ai-tool)
- [Tool surface](#tool-surface)
- [For contributors — extending the server](#for-contributors--extending-the-server)
- [Design rationale](#design-rationale)

## At a glance

```
   ┌──────────────────────────────┐         ┌────────────────────────────┐
   │ Your AI tool (any device)    │         │ Your browser (any device)  │
   │  Claude Code / Codex /       │         │  http://lerobot.local/     │
   │  Cursor / Claude Desktop /   │         │                            │
   │  custom scripts …            │         │  GUI tab subscribes to     │
   │                              │         │  bridge for navigate /     │
   └──────────────┬───────────────┘         │  notify commands           │
                  │                         └─────────────┬──────────────┘
                  │ HTTPS + bearer                        │ WebSocket
                  ▼                                       ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ robot host (lerobot.local) — ONE process, port 8000                   │
   │   GET  /              → GUI                                           │
   │   GET  /ai_setup      → token-issuance page                           │
   │   POST /mcp           → MCP HTTP transport (bearer-token gated)       │
   │   WS   /api/bridge/ws → bridge subscriber channel                     │
   └───────────────────────────────────────────────────────────────────────┘
```

One service runs on the robot host (`lerobot-gui`). It serves the GUI,
the AI-setup page, the bridge channel, and the MCP HTTP transport on
one port. Each user, on their own device, registers their AI tool
against the host with a one-line copy-paste command from
`lerobot.local/ai_setup`.

<p align="center">
  <img alt="End-to-end demo: an AI tool browsing datasets and driving the GUI" src="docs/demo_e2e.gif" width="90%">
</p>

---

## For operators — connecting your AI tool

### Prerequisites

- **A working LeRobot install on the robot host.** Assumes the GUI
  already runs and Python deps are set up. See the project root README
  and `docs/source/` for the broader install.
- The `lerobot[mcp]` and `lerobot[gui]` extras installed:
  ```bash
  uv sync --extra mcp --extra gui --extra dataset
  ```
- An MCP-aware AI tool on each user device: Claude Code, Codex CLI,
  Cursor, Claude Desktop, or any client that speaks MCP.
- A LAN where `*.local` mDNS resolves (most home / office networks do).

### Host setup (one-time, by whoever maintains the robot host)

One command brings up the GUI + MCP endpoint + AI setup page on the
same port (`8000` by default):

```bash
lerobot-gui --host 0.0.0.0 --port 8000
```

The startup banner prints all the URLs your LAN can reach. Bound to
`0.0.0.0` (not loopback) so other machines on the network can reach
it; mDNS auto-publishes `lerobot.local` so they don't need your IP.

To survive reboots (optional), save a systemd unit such as:

```ini
[Unit]
Description=LeRobot host service (GUI + MCP, LAN-discoverable via mDNS)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lerobot
Group=lerobot
WorkingDirectory=/opt/lerobot
Environment=HF_LEROBOT_HOME=/var/lib/lerobot/datasets
ExecStart=/opt/lerobot/.venv/bin/lerobot-gui --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=read-only

[Install]
WantedBy=multi-user.target
```

as `/etc/systemd/system/lerobot-gui.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lerobot-gui.service
journalctl -u lerobot-gui -f      # follow logs
```

Skip this entirely if you don't need it — `lerobot-gui` in a terminal
or tmux is enough for a typical lab. No admin password, no AI account,
no project secrets — just one service.

### User setup (per person, per device, ~90 seconds)

Done from any browser on the same LAN.

**1. Open the AI setup page.** Visit
`http://lerobot.local/ai_setup`. If your network doesn't resolve
`lerobot.local`, use the host's IP.

**2. Issue a token for this device.** Pick a name (`alice-laptop`,
`bob-cursor`) and the scopes you want — see the [Scope
hierarchy](#scope-hierarchy) section below. Click **Generate token**.

The next screen shows the bearer **once** plus a copy-paste command
for each supported AI tool (Claude Code, Codex CLI, Gemini CLI,
Cursor, Claude Desktop). The bearer is not recoverable — save it now
(or re-issue later if you lose it).

<p align="center">
  <img alt="The /ai_setup token-issuance page with per-tool registration snippets" src="../gui/docs/images/ai_setup_expanded.png" width="80%">
</p>

**3. Register with your AI tool.** Copy the relevant command. For
Claude Code:

```bash
claude mcp add --transport http lerobot \
  http://lerobot.local:8000/mcp \
  --header 'Authorization: Bearer sk-lr-XXXXXXXX'
```

Verify:

```bash
claude mcp list
# expect: lerobot — connected
```

**4. (Optional) Enable browser notifications.** When you ask the AI
to watch a long-running task, the `notify_user` command uses the OS
notification banner so you can switch windows. First fire prompts the
browser for permission; accept it. Until then, the GUI falls back to
flashing the tab title.

### Verify end-to-end

Open `http://lerobot.local` in a browser (keep the tab open). In your
AI tool, ask:

> _"What lerobot datasets do I have? Pick the most interesting one."_

You should see the AI list your datasets and pick one. Now:

> _"Open episode 0 of that dataset in my GUI."_

The already-open GUI tab navigates to the episode viewer. Bridge tools
like `highlight_in_viewer` then let the AI mark items in that same tab:

<p align="center">
  <img alt="highlight_in_viewer: the AI highlighting episodes in the open GUI tab" src="docs/proofs/feat_highlight_viewer.png" width="80%">
</p>

### Try these next

- _"Show me the last frame of episodes 0–9 of `<dataset>` and tell me
  which look like the task succeeded."_
- _"Tag the ones that look successful with `outcome=success`."_
- _"Close this conversation. Tomorrow ask: what did we conclude
  yesterday?"_ — annotations survive in the host's sidecar.
- _"Highlight failed episodes in my GUI viewer."_
- _"Start a 10-episode recording with task='cylinder ring assembly'.
  Let me know when it's done."_ (requires `operate`)

### Common operations

**See active tokens.** Run `lerobot-mcp list-tokens` on the host, or
visit `lerobot.local/ai_setup` — the listing is on the same page.

**Revoke a token.** Click **Revoke** next to the entry on
`/ai_setup`, or `lerobot-mcp revoke-token --name alice-laptop` on the
host. The client tool's calls start failing with 401 immediately.

**Re-issue a token.** Revoke first (or pick a new name), then
re-issue. Tokens are one-shot secrets — there's no "show me the token
again."

**Multiple robot hosts on the same LAN.** Add one MCP entry per host; the AI
sees them as separate tool namespaces.

**When the robot host moves (different IP).** If you registered with
the `lerobot.local` hostname (the `/ai_setup` default), nothing
changes — mDNS auto-resolves to the new IP. Hardcoded IPs need
re-registration; always prefer the hostname form.

**Updating after a code change.**

```bash
cd /opt/lerobot   # or wherever lerobot is installed on the host
git pull
uv sync --extra mcp --extra gui   # if deps changed
sudo systemctl restart lerobot-gui   # or restart your foreground run
```

User devices don't need any action — their tokens keep working.

### Troubleshooting

| Symptom                                      | Likely cause                                                  | Fix                                                                            |
| -------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `claude mcp list` shows lerobot disconnected | GUI not running, or URL/token typo                            | check `ss -tlnp \| grep 8000`; re-check the URL+token in the registration      |
| Tool calls return 401                        | token revoked or wrong                                        | re-issue from `/ai_setup`                                                      |
| Tool calls return 403                        | token lacks the required scope                                | re-issue with the needed scope (`operate` for motion / starting runs)          |
| `navigate_to` returns `delivered: 0`         | no GUI tab is open right now                                  | open the GUI; the bridge needs at least one subscribed tab                     |
| Notifications never appear                   | permission denied in browser, or first-load prompt was missed | check `Notification.permission` in DevTools; re-grant in browser site settings |
| AI tool can't reach `lerobot.local`          | mDNS not propagating across the network                       | use the host's IP for the URL, or fix mDNS at the network level                |

GUI logs are wherever you started the GUI (or `journalctl -u
lerobot-gui` if you set up the systemd unit).

---

## Tool surface

### Scope hierarchy

Strict superset, **token must carry every scope it needs**:

| Scope     | Grants                                                                                                | Typical token                                             |
| --------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `read`    | List / read datasets, episodes, frames, models, robots, runs, Hub status. No state changes.           | Default for any LAN client                                |
| `comment` | All of `read`, plus write / delete sidecar comments. Never touches canonical files.                   | Read-only-with-findings assistants                        |
| `edit`    | All of `comment`, plus mutate canonical state: dataset edits, profile CRUD, Hub uploads. No hardware. | Trusted dev tokens                                        |
| `operate` | All of `edit`, plus hardware-moving operations: motors, recording, training, recovery.                | Granted explicitly per device, human near the kill switch |

`auth.py` exposes `SCOPE_READ`, `SCOPE_COMMENT`, `SCOPE_EDIT`,
`SCOPE_OPERATE` constants. `ALL_SCOPES` is the canonical tuple.

Legacy tokens issued before the rename (with `annotate` scope) keep
working — `_canonicalize_stored_scopes()` maps `annotate` → `comment`
on read; `_validate_scopes()` accepts the legacy name on issuance and
stores it as `comment`. The migration is invisible to callers.

### Shipped today

| Domain      | Tool                                                                                                                                                                                                       | Scope     |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Bridge / UI | `navigate_to`, `notify_user`, `highlight_in_viewer`                                                                                                                                                        | `read`    |
| Dataset     | `list_datasets`, `get_dataset_info`, `list_episodes`, `get_episode_summary`, `get_frame`, `get_episode_tags`, `list_pending_edits`, `list_tagged_episodes`, `get_feature_series`, `validate_dataset_merge` | `read`    |
| Dataset     | `tag_episode`, `delete_episode_tag`                                                                                                                                                                        | `comment` |
| Dataset     | `propose_set_feature`, `propose_delete_episode`, `propose_trim_episode`, `discard_pending_edits`, `apply_pending_edits`, `merge_into_dataset`                                                              | `edit`    |
| Hub         | `hub_auth_status`, `hub_repo_info`, `hub_list_jobs`, `hub_job_progress`, `hub_diff_local_vs_remote`                                                                                                        | `read`    |
| Hub         | `hub_start_upload`, `hub_start_download`, `hub_cancel_job`                                                                                                                                                 | `edit`    |
| Run         | `get_run_status`, `get_run_output`, `get_latency_metrics`, `get_rlt_metrics`                                                                                                                               | `read`    |
| Robots      | `list_robot_profiles`, `get_robot_profile`, `list_teleop_profiles`, `get_teleop_profile`, `list_ports`, `get_all_port_assignments`                                                                         | `read`    |
| Robots      | `create/update/rename/delete_robot_profile`, `create/update/rename/delete_teleop_profile`, `assign_port_to_arm`                                                                                            | `edit`    |
| Introspect  | `lerobot_whoami`, `lerobot_list_tools`                                                                                                                                                                     | `read`    |

### Designed but not yet shipped

The MCP surface the design calls for, grouped by domain and scope.
Reviewers cross-check against the implementation when extending. Names
may shift slightly on landing.

| Domain      | Tool                                                                                                                                                             | Scope     |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Bridge / UI | `set_filter` (waits for a filter UI to land — see `gui/TODO.md`)                                                                                                 | `read`    |
| Dataset     | `delete_dataset` (no GUI feature yet, deferred)                                                                                                                  | `edit`    |
| Models      | `list_model_sources`, `list_models_in_source`, `list_run_checkpoints`, `get_run_config`                                                                          | `read`    |
| Models      | `add_model_source`, `remove_model_source`                                                                                                                        | `edit`    |
| Models      | `load_debug_model`, `unload_debug_model`, `get_debug_status`                                                                                                     | `operate` |
| Robots      | `identify_arm`, `detect_cameras_into_profile`, `start/finish/cancel_rest_recording`, `move_to_rest_position`, `record/replay/delete_trajectory`, `recover_robot` | `operate` |
| Run         | `update_rlt_config` (parked — PR #27, narrow scope)                                                                                                              | `edit`    |
| Run         | `start_teleoperate`, `start_record`, `start_replay`, `start_hvla`, `stop_current_run`                                                                            | `operate` |

Design conventions worth flagging:

- The `propose_*` / `apply_*` split mirrors the GUI's `PendingEdit`
  model: each `propose_*` adds to a per-dataset staging queue;
  `apply_pending_edits` locks the dataset and writes through to
  parquet on disk; `discard_pending_edits` drops the queue without
  touching disk.
- Hub ops are async — `_start_*` tools return a `job_id` immediately;
  the AI polls `hub_list_jobs` / `hub_job_progress(job_id)` to track
  completion.
- `stop_current_run` is the universal kill-switch for `operate` —
  always allowed at scope, no confirmation. There is no destructive
  counterpart; stop is always safe.

---

## For contributors — extending the server

### Layout

```
src/lerobot/mcp/
├── README.md          ← you are here
├── __init__.py
├── auth.py            ← TokenStore, scopes, requires_scope decorator
├── bridge_tools.py    ← AI → GUI bridge tools (navigate, notify, highlight)
├── cli.py             ← `lerobot-mcp` entry point (stdio + http transports)
├── server.py          ← `build_server()` + dataset / comment tools
├── store.py           ← AnnotationStore (sidecar SQLite)
└── docs/
    ├── demo_e2e.{py,gif,mp4,md}    ← end-to-end demo of the v1 surface
    └── proofs/feat_*.png           ← per-feature visible-effect proofs
```

The GUI side lives at `src/lerobot/gui/api/bridge.py` (FastAPI
dispatch and WebSocket subscribers) and `src/lerobot/gui/static/bridge.js`
plus `bridge_consumers.js` (browser-side handler).

### Adding a new tool

**1. Pick the scope.**

- Reads only? → `SCOPE_READ`
- Writes to the sidecar (notes, tags, comments)? → `SCOPE_COMMENT`
- Mutates canonical on-host state (dataset / profile / Hub)? → `SCOPE_EDIT`
- Starts hardware or long-running jobs? → `SCOPE_OPERATE`

When unsure, lean down a tier and pull up only if you genuinely need
mutation power. A `read`-tier tool that turned out to also write
would get rejected at scope-check time — clear failure, easy fix. The
opposite mistake (over-granting) is silent.

**2. Write the function.** In `server.py`, inside `build_server`:

```python
@mcp.tool()
@requires_scope(SCOPE_COMMENT)
async def my_new_tool(repo_id: str, key: str) -> dict[str, Any]:
    """One-line description shown to the AI.

    Longer description optional. Always document return shape.

    Args:
        repo_id: Dataset id (logical, never a path).
        key: ...

    Returns:
        ``{...}`` — what the AI gets back.
    """
    s = _state()                      # dataset_root + annotation store
    meta = s.get_meta(repo_id)        # raises if dataset missing
    ...
    return {"repo_id": repo_id, "...": ...}
```

Conventions worth keeping:

- **Async** unless the tool truly has no I/O. FastMCP runs sync tools
  on the same event loop, blocking it; async tools `await` cleanly.
  Bridge tools are all `async def`.
- **Logical IDs only.** Datasets are `repo_id`, episodes are
  `episode_id`, runs are `run_id`. Never accept or return a path. The
  server resolves IDs internally.
- **Validate at the boundary.** `meta = s.get_meta(repo_id)` raises
  if the dataset doesn't exist. Episode-index bounds get checked
  explicitly. Don't trust caller input. **Use `raise ValueError(...)`,
  never `assert`** — assertions disappear under `python -O` and
  produce ugly tracebacks the AI can't parse.
- **Docstrings ARE the prompt.** The AI sees the docstring when it
  picks which tool to call. Lead with what the tool does, then how.
- **Param names anchor the semantics.** This matters because the AI
  reads the schema, not just the docstring. Prefer names that close
  off ambiguous readings — `keep_from` / `keep_to` over
  `start_frame` / `end_frame` (which could be misread as "the range
  to remove"). When the FastAPI surface uses different names for
  backwards compat, translate at the MCP wrapper, don't propagate
  the ambiguity.
- **Make the response outcome-transparent.** The AI sees the
  response and uses it to decide what to do next. The response
  message should describe what actually happened in terms the agent
  can self-correct from. Echo back the numbers that matter:
  `"keep 6 of 770 frames; dropping 764"` is far more useful than
  `"trim staged"`. For overwrites, include `previous_value` /
  `overwrote: true`. For batch ops, include counts.
- **Conflict-vs-validation split.** Hard failures (out of bounds,
  unknown id, schema violation) raise `ValueError` → the AI sees
  `isError=True` and the message. Recoverable conflicts (overlap
  with prior state, large-batch threshold) return a structured
  `{"status": "conflict", "detail": {"code": "...", ...}}` dict so
  the AI can read `detail.code` and retry with the right confirm
  flag rather than parsing error strings.
- **Return shape is part of the contract.** Always a dict (or an MCP
  `Image` for binary content). Keep keys stable across versions;
  additive changes (new fields) are safe, key removals/renames are
  breaking.
- **Thin wrapper around a unit-tested core.** When the same logic
  also serves a FastAPI route (or any other surface), extract a
  pure-Python helper that takes `AppState` + validated params and
  raises typed exceptions; both surfaces wrap it. The MCP tool body
  should be ~5-10 lines: validate scope, resolve state, call helper,
  translate exceptions. **Do not auto-bind FastAPI routes to MCP
  tools.** See the anti-patterns section.

**3. Test it.** Add a unit test in `tests/mcp/test_<domain>.py`. Use
the existing fixtures in `tests/mcp/` — they synthesize a minimal
in-memory dataset on `tmp_path`, no network, no real datasets. The
MCP test pattern (see `tests/mcp/test_bridge.py` /
`test_dataset_edit.py` for examples):

```python
def test_my_new_tool() -> None:
    mcp = build_server(...)
    async def run():
        return await mcp.call_tool("my_new_tool", {"repo_id": "...", "key": "..."})
    _, structured = asyncio.run(run())
    assert structured["..."] == ...
```

Coverage bar — for every tool, the test file should include:

- **Happy path** — verifies the response shape and state side-effect.
- **One test per error case** the docstring promises (out-of-range,
  unknown id, duplicate, schema violation, ...). Use
  `pytest.raises(Exception, match="<message fragment>")` for hard
  failures, or assert `result["status"] == "conflict"` for
  recoverable ones.
- **End-to-end test for destructive tools** — for `edit` / `operate`
  tools that mutate disk or hardware, drive the full propose →
  apply → verify-on-disk pipeline against a synthetic dataset in
  `tmp_path`. See `TestApply::test_propose_then_apply_deletes_episode_from_disk`
  for the pattern.
- **NEVER point a test or proof at the user's real datasets.**
  Always use the synthetic factory or a clearly-marked throwaway
  (e.g. `thewisp/test_*_do_not_use`).

**4. Add visible proof.** For any tool with a user-observable effect,
generate a **transcript log** — a markdown of the call shape + response
shape, including happy paths AND error cases AND any "confused agent"
scenario where input semantics could be misread. See
`mcp/docs/proofs/dataset_edit_transcript.md` (or any of the other
`*_transcript.md` files) for the format. For tools that change what the
GUI shows on-screen, also capture a before/after PNG pair (e.g. the
`feat_*.png` files alongside the transcripts).

The driver scripts that produced these transcripts/PNGs are
**not** checked in — they're hardcoded to the author's local host and
throwaway repos and nobody else can run them. The transcripts and PNGs
**are** the durable artifacts; treat them as the authoritative record
of what the tool did at merge time.

Wire-level error response shape (`isError=True, content=[message]`)
should also be probed at least once during development against the
unified GUI's `/mcp/` endpoint. It doesn't need a formal test, but
the PR body should cite the actual response strings the AI sees.

**5. Update the tool surface table.** Add the tool to the [Shipped
today](#shipped-today) table above (and remove from the [Designed but
not yet shipped](#designed-but-not-yet-shipped) table if it was listed
there). This is the human-readable index of what's available + at
what scope.

### Adding a new bridge command type

Bridge commands flow:

```
MCP tool (bridge_tools.py)
  └─ await _dispatch(type, params)
       ├─ unified mode → in-process await registry.dispatch(...)
       └─ split-process mode → urllib POST to /api/bridge/_dispatch
            └─ FastAPI dispatch endpoint (gui/api/bridge.py)
                 └─ WS broadcast to subscribers
                      └─ bridge.js receives, fires CustomEvent
                           └─ bridge_consumers.js handler
                                └─ real GUI side effect (openDataset, etc.)
```

A new command type needs **four** matching changes:

1. **MCP tool** in `mcp/bridge_tools.py` — `async def` returning
   `await _dispatch("my_type", params)`. Scope it `SCOPE_READ`
   (bridge commands change what the user sees, not canonical state).
2. **Wire allowlist** in `gui/api/bridge.py` — add `"my_type"` to
   `SUPPORTED_COMMAND_TYPES`. The frozenset is the protocol's source
   of truth.
3. **Browser dispatcher** in `gui/static/bridge.js` — add a `case "my_type"` branch in the message handler. The minimum is dispatching `new CustomEvent("lerobot-bridge:my_type", {detail: params})`; bridge.js stays feature-agnostic.
4. **Consumer** in `gui/static/bridge_consumers.js` — listen for the
   custom event, reach into existing app.js primitives to make the
   effect happen. **Integrate with renderTree state** (don't just
   mutate DOM directly) so the effect survives re-renders.

Then capture a before/after PNG pair of the visible effect (alongside
the transcript) so the bridge behavior is verified.

### Anti-patterns to avoid

- **Don't auto-bind FastAPI routes to MCP tools.** Wire each tool
  explicitly. Scope, granularity, and AI-friendly docstrings differ
  from HTTP route conventions; auto-binding leaks GUI internals.
- **Don't add a tool with no visible effect** if the design claims
  one (the v1 `set_filter` lesson — advertised but had no
  `dataset-search` UI to land on). Either land the GUI surface in the
  same PR, or don't ship the tool. The per-feature proof pattern is
  a before/after PNG pair alongside the transcript (see
  `mcp/docs/proofs/feat_*.png`).
- **Don't share state via module globals** between two `build_server`
  instances. `ServerState` is bound to a single instance via closure
  capture; if you find yourself wanting `global state`, refactor to
  pass it through.
- **Don't widen the deployment-mode gap in bridge return shapes.**
  Today the in-process success path returns
  `{"delivered": N, "type": ..., "client_id": ...}` and the HTTP
  fallback's failure path returns `{"delivered": 0, "reason": ...}`.
  They're already not identical (success vs failure shapes carry
  different keys), but the gap is bounded. When adding new fields,
  add them to BOTH paths' success shapes so an agent's response
  parser doesn't have to branch on deployment mode.

### Common patterns

- **Dataset → meta lookup**: `s = _state(); meta = s.get_meta(repo_id)`
  is the canonical "resolve a dataset by id" pattern. `meta` is a
  `LeRobotDatasetMetadata`.
- **Sidecar read/write**: `s.store.set_tag(...)`, `s.store.get_tags(...)`,
  `s.store.delete_tag(...)`. All path-safe (parameterized SQL).
- **Bounds-checked episode access**: `if not (0 <= episode_id < meta.total_episodes): raise ValueError(...)`. Mirror this in every episode-scoped tool.

### Demos

The shipped demo artifacts live under `mcp/docs/`:

- `demo_e2e.md` / `demo_e2e.gif` / `demo_e2e.mp4` — end-to-end 9-call
  walkthrough against the GUI.
- `proofs/*_transcript.md` — per-tool call/response logs (read-tier,
  dataset edit, hub edit, robots, run).
- `proofs/feat_*.png` — before/after screenshots for the bridge tools
  that change what the GUI shows.

These are static snapshots of "what worked at this commit," not
regression tests — the formal coverage is in `tests/mcp/`. When
adding or changing a tool, re-capture the artifact for that tool
(transcript or PNG pair) and check it in alongside the code.

---

## Design rationale

The motivation and choices behind the shape above. Read this if
you're reviewing the design, planning a successor feature, or
wondering "why isn't there a chat panel in the GUI?".

### Goal

Any user on the LAN, using their preferred AI tool with their
existing subscription, can drive LeRobot through the MCP server. The
LeRobot GUI runs in a browser tab alongside; AI **bridge tools**
(navigate the GUI, notify the user, highlight items in view) keep the
GUI in sync with whatever the AI is helping the user explore — so the
user mostly stays in their AI tool, glances at the GUI when the AI
raises something, and never has to context-switch by hand.

### Non-goals

- **No in-GUI AI chat panel.** Subscription-based OAuth for
  third-party browser apps is not generally available — Anthropic /
  OpenAI / Google reserve it for their own first-party tools. Asking
  users to obtain a separate paid API account just to chat with their
  robot data is the wrong tradeoff. We bring lerobot to the user's
  existing tool instead. Revisit if/when major providers open
  third-party OAuth, or a user community emerges that explicitly
  wants the API-key path.
- **No host-level auth layer introduced by AI-native.** Whoever can
  administer the robot host (install, restart, SSH) is implicitly the
  trusted operator. `/ai_setup` is gated by LAN-trust, not by a
  separate password.
- **No "LeRobot project AI account."** The project never pays anyone's
  LLM bill, never holds anyone's data, never picks model providers.
  Every LLM call is authenticated by the end user (BYOK all the way
  down).
- **No mobile-specific UX.** Tablets are reasonable when a teleoperator
  view eventually needs one; phones are not a target. Assume a
  desktop browser; do not design for QR codes, camera access,
  accelerometer input, or PWA installation.
- **No multi-host coordination.** Out of scope: an AI tool on host A
  reading host B's data, training distributed across hosts, shared
  conversations across hosts, federated identity, fleet-wide deploys.
  Moving things between hosts is via HF Hub push/pull, the same as
  today. _Not_ a constraint against multiple _independent_ hosts on
  one LAN — each publishes its own mDNS name; the user adds one MCP
  entry per host.
- **Open ecosystem, not vendor-neutral protocol.** MCP was initiated
  by Anthropic; the standard is open and adopted by OpenAI / Google
  tooling, but it's not vendor-blind. The principle to hold: lerobot
  ships an MCP server, and any MCP client (whatever its model or auth
  backend) can use it.
- **No agent filesystem access.** Agents never see paths, never read
  arbitrary files, never `exec` shell commands. Every action is a
  typed MCP tool. `read_file` / `write_file` / `bash` tool families
  are out of scope — too easy to break out of the typed surface, and
  the underlying use cases are better served by dedicated MCP tools.
- **No persistent agent across user sessions.** "AI keeps working
  when I close my laptop" isn't a goal. The user's AI tool is the
  runtime; if it's closed, the agent is closed. Long-running watchers
  run in the user's environment for as long as their tool stays
  open — acceptable for the research-lab use case the design
  optimizes for.

### Architectural decisions

1. **MCP server lives on the robot host.** Every client — any AI tool
   the user picks, scheduled routines, batch scripts — connects to
   the same MCP service. One MCP server per host, not one per client.

2. **The agent loop runs in the user's tool of choice.** We do not
   host an LLM agent loop. The user's Claude Code / Codex / Cursor /
   etc. is the agent runtime; lerobot is just a tool source. Their
   existing subscription pays for LLM calls; the LLM credential never
   reaches the robot host.

3. **Tools take logical IDs, never file paths.** Datasets are
   identified by `repo_id`, runs by `run_id`, episodes by
   `episode_id`. The agent never receives a path in tool output and
   never passes one as input. The server resolves IDs to paths
   internally. This includes local-only artifacts: a recording-only
   dataset still has a `repo_id` derived from its `$HF_LEROBOT_HOME`
   location, with `is_local_only` / `hub_status` metadata on the
   entity.

4. **The agent's authority matches the supervision it's under** —
   the 4-tier `read` ⊂ `comment` ⊂ `edit` ⊂ `operate` hierarchy
   above. Each MCP token is issued with an explicit scope set; each
   tool declares its required scope; middleware rejects calls whose
   token scope is not a superset of the tool's required scope.

   **Why four tiers, not three.** Splitting `edit` (data + config
   mutations) from `operate` (hardware-moving + long-running) lets a
   user grant "clean up datasets, push to Hub" without also granting
   "start the arm." Squashing them would mean a "let the AI tidy
   data" token can also drive motors — wrong by default.

   **Confirmation handled by the AI tool, not by the protocol.** No
   protocol-level two-phase commit; reflects industry practice
   (GitHub, GDocs, Calendar, Notion all do direct API calls with
   scope-based auth + reversibility, no per-call confirm step).
   Dangerous-by-default tools surface a `DESTRUCTIVE:` prefix in
   their docstring so the AI tool's "approve this call" UI highlights
   them; reversibility is designed in via (a) the `propose_*` +
   `apply_*` staging pattern for dataset edits, mirroring the GUI's
   `PendingEdit` model; (b) version history / git for Hub-pushed
   datasets; (c) the unconditional `stop_current_run` kill switch for
   `operate`. Undo / redo across the sidecar is intentionally out of
   scope at this layer.

5. **The AI drives the GUI via _bridge tools_, not by being inside
   it.** A small family of MCP tools (`navigate_to`, `notify_user`,
   `highlight_in_viewer`, future `set_filter`, ...) lets the AI cause
   something to happen in the user's open GUI tab. Implementation:
   the tool writes a command to a per-session queue; the GUI
   subscribes via WebSocket and acts on it.

6. **The GUI uses the Web Notifications API** to refocus a
   backgrounded tab when the AI asks. Standard pattern used by
   Slack, Linear, Gmail, GitHub. Permission is requested lazily —
   only when the user does something the feature actually helps with
   (e.g. asks the AI to watch a long-running task).

7. **Sidecar SQLite for agent-written comments**, never writes to the
   canonical dataset. Shared across all AI tools that connect to one
   host. The `comment` scope grants write access; the sidecar
   persists across AI sessions and AI tools so a comment Claude
   leaves today is visible to Codex next week to `alice` running her
   own AI assistant. Comments are **eventually surfaced in the GUI
   too** (sidebar / annotations panel — see `gui/TODO.md`); today
   they're MCP-only.

8. **Isolation model — bridge commands are scoped, everything else
   isolates naturally.** Even though multi-user is out of scope, the
   bridge protocol carries identity from day one so it doesn't have
   to be retrofitted later.

   | Category            | Examples                                                          | Isolation behaviour                                                                                                                                                                                            |
   | ------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | Read operations     | `list_datasets`, `get_frame`, `find_episodes`                     | **Naturally isolated.** Returns flow back via the JSON-RPC response to the calling AI tool only. User A's query never reaches user B.                                                                          |
   | Sidecar annotations | `tag_episode`, `get_episode_tags`                                 | **Intentionally shared.** The whole point is a "shared brain in writing" across all AI tools on this host. User A's tags are readable by user B. Authorship can be recorded later if/when attribution matters. |
   | Hardware-mutating   | `start_recording`, `start_training_run`, motor ops                | **Inherently shared.** There is one arm and one GPU; activity is necessarily visible to everyone.                                                                                                              |
   | **Bridge commands** | `navigate_to`, `notify_user`, `highlight_in_viewer`, `set_filter` | **Must be scoped to the originator's GUI tab.** Otherwise user A asking "open episode 47" would yank user B's view around.                                                                                     |

   Single-operator (today's default) is "every tab defaults to
   wildcard, everything just works." Multi-user labs will set their
   GUI tab to their own `client_id` via a one-click action on
   `/ai_setup` that stamps a cookie. The protocol carries the field
   from day one; multi-user becomes a UI affordance, not a rebuild.

### Operational lifecycle

How the working setup changes over time:

| Operation                                  | What happens / what the user does                                                                                                                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Add a host's MCP to an AI tool**         | Visit the host's `/ai_setup`, name a device, copy-paste the generated command.                                                                                                                                                                   |
| **Remove a host's MCP from an AI tool**    | Two ends, independent: revoke server-side via `/ai_setup`'s Revoke button; remove client-side via `claude mcp remove lerobot` (or equivalent). Revoking nukes server-side access immediately; the client config stays dangling until cleaned up. |
| **Robot host moves (different IP)**        | If the user registered the mDNS-hostname form (`http://lerobot.local:8000/mcp`, the `/ai_setup` default), nothing changes — mDNS resolves to the new IP automatically. Hardcoded IPs don't survive moves and must be re-registered.              |
| **Robot host renamed**                     | Re-register on the new name. mDNS naming collisions auto-suffix (`lerobot-2.local`); be deliberate when running multiple hosts.                                                                                                                  |
| **Token rotated / accidentally exposed**   | Admin revokes via `/ai_setup`; user generates a new one + updates their AI tool. Calls under the old token start failing with 401 — clear failure mode.                                                                                          |
| **Multiple independent robots on one LAN** | Each host runs its own MCP service under a distinct mDNS name (e.g. `--base-name fleet-001`). User adds one MCP entry per host: `lerobot-bench`, `lerobot-prod`, ... The AI sees them as separate tool namespaces.                               |
| **AI tool upgrade / reinstall**            | Tokens stay valid; user re-pastes the registration line into the new tool config. No server-side change.                                                                                                                                         |

The mDNS-hostname convention is what makes most of this graceful.
The design deliberately tells `/ai_setup` to emit the hostname form
so users don't have to think about any of this until they explicitly
opt out (e.g. for an off-LAN deployment).

### Open decisions

1. **Bridge command queue: in-memory or SQLite-backed?** In-memory
   simpler; SQLite survives GUI/MCP restarts but adds little for
   commands that are inherently transient (a navigate command from
   yesterday doesn't matter today). **Lean: in-memory, with the
   queue bounded per session.**
2. **Single-operator vs named-identity GUI sessions.** Today: every
   GUI tab on the LAN receives every bridge command (subscribers
   declare their target on `/api/bridge/ws` with no auth; `*` is the
   default). A LAN actor can subscribe wildcard and tail every
   navigate / notify the operator's AI fires. Mitigation is upstream
   of multi-user: either drop the LAN-trust posture for
   `/api/bridge/ws` (require a bearer on WS upgrade — and then the
   GUI tab needs a token), or default tabs to a per-tab UUID target
   so wildcard subscription becomes an explicit operator action.
   **Lean: defer to a follow-up PR; ship single-operator on
   LAN-trust now.** Tracked in `gui/TODO.md`.
3. **Notification permission UX.** Browsers (Firefox especially)
   demote sites that prompt on first load. **Lean: ask only when the
   user does something that benefits from notifications** (e.g. asks
   the AI to watch a long-running task), with a one-click "enable
   notifications" prompt in the GUI's settings as a manual escape.
4. **Token format.** Opaque random string vs JWT. **Lean: opaque,
   server-side lookup** (already implemented; revisit only if there's
   pressure).
