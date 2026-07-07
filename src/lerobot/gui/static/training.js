// Copyright 2026 The HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// Training panel — frontend for the training API.
// Lives in the Model tab (sidebar + main pane).
//
// What this drives:
// - GET /api/training/hosts                      → host info display
// - GET /api/training/runs                       → run list + active polling
// - POST /api/training/runs                      → start a run
// - GET /api/training/runs/{id}                  → detail (progress, checkpoints, log)
// - POST /api/training/runs/{id}/stop            → user-initiated stop
// - GET /api/datasets/sources + .../datasets     → populate the dataset dropdown

const TRAINING_POLL_MS = 3000;
let _trainingHosts = [];

// ── Nebius connection (server-held service-account credential) ──────────────
// One service-account key for the whole GUI server, configured once via the
// Nebius-connection modal. The server stores the key (0600) and uses it for
// every ephemeral spawn/teardown; the browser only sees non-secret status.
// Shared by anyone who can reach the server — same trust model as the
// server's HF token / SSH key. See DESIGN.md § Authentication.
let _nebiusConnStatus = null;  // last fetched status, or null until loaded
async function trainingFetchNebiusConnection() {
  try {
    const resp = await fetch("/api/training/nebius/connection");
    _nebiusConnStatus = resp.ok ? await resp.json() : null;
  } catch { _nebiusConnStatus = null; }
  return _nebiusConnStatus;
}
function _nebiusConnSummary(st) {
  if (!st) return "Status unknown.";
  if (st.configured) {
    const sa = st.service_account_id ? ` (${st.service_account_id})` : "";
    return `✓ Connected${sa} · project ${st.project_id} · subnet ${st.subnet_id}`;
  }
  if (st.has_key) return "⚠ Key present but project/subnet missing.";
  return "✗ Not connected — paste a service-account key to enable cloud spawning.";
}
function trainingOpenNebiusConnection() {
  const overlay = document.getElementById("nebius-conn-overlay");
  if (!overlay) return;
  document.getElementById("nebius-conn-status").textContent = "";
  // Reset to the default (CLI) mode and clear any previously-typed secrets.
  const modeSel = document.getElementById("nebius-conn-mode");
  if (modeSel) modeSel.value = "json";
  ["nebius-conn-key", "nebius-conn-privkey", "nebius-conn-kid", "nebius-conn-said"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  trainingNebiusModeChanged();
  const foundSel = document.getElementById("nebius-conn-subnet-found");
  if (foundSel) { foundSel.style.display = "none"; foundSel.innerHTML = ""; }
  const subnetHint = document.getElementById("nebius-conn-subnet-hint");
  if (subnetHint) subnetHint.textContent = "";
  overlay.style.display = "flex";
  trainingFetchNebiusConnection().then((st) => {
    document.getElementById("nebius-conn-current").textContent = _nebiusConnSummary(st);
    // Pre-fill the non-secret ids so a re-save doesn't force re-typing them;
    // the key field stays empty (never echoed back).
    if (st) {
      document.getElementById("nebius-conn-project").value = st.project_id || "";
      document.getElementById("nebius-conn-subnet").value = st.subnet_id || "";
    }
  });
}
function trainingCloseNebiusConnection() {
  const overlay = document.getElementById("nebius-conn-overlay");
  if (overlay) overlay.style.display = "none";
  // Don't retain pasted secrets in the DOM (either input style).
  ["nebius-conn-key", "nebius-conn-privkey"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
}
// Toggle the two key-entry layouts: a pasted key file (CLI) vs the private
// key + IDs (console). The server assembles the same JSON from either.
function trainingNebiusModeChanged() {
  const sel = document.getElementById("nebius-conn-mode");
  const mode = sel ? sel.value : "json";
  const jsonEl = document.getElementById("nebius-conn-json-fields");
  const fieldsEl = document.getElementById("nebius-conn-fields-fields");
  if (jsonEl) jsonEl.style.display = mode === "json" ? "block" : "none";
  if (fieldsEl) fieldsEl.style.display = mode === "fields" ? "block" : "none";
}
// The key portion of the payload for the active mode, or null (after writing a
// message to statusEl, if given) when required fields are missing. Shared by
// Connect and the subnet-discovery lookup.
function _nebiusKeyPayload(statusEl) {
  const sel = document.getElementById("nebius-conn-mode");
  const mode = sel ? sel.value : "json";
  if (mode === "json") {
    const key_json = document.getElementById("nebius-conn-key").value.trim();
    if (!key_json) {
      if (statusEl) statusEl.textContent = "Paste the service-account key JSON.";
      return null;
    }
    return { key_json };
  }
  const private_key = document.getElementById("nebius-conn-privkey").value.trim();
  const key_id = document.getElementById("nebius-conn-kid").value.trim();
  const service_account_id = document.getElementById("nebius-conn-said").value.trim();
  if (!private_key || !key_id || !service_account_id) {
    if (statusEl) statusEl.textContent = "Paste the private key and both IDs (key ID + service account ID).";
    return null;
  }
  return { private_key, key_id, service_account_id };
}
// Look up the project's subnets so the operator can pick instead of hand-copying
// an ID. Best-effort: on any failure the manual Subnet field still works.
async function trainingFindSubnets() {
  const hint = document.getElementById("nebius-conn-subnet-hint");
  const select = document.getElementById("nebius-conn-subnet-found");
  const input = document.getElementById("nebius-conn-subnet");
  const project_id = document.getElementById("nebius-conn-project").value.trim();
  const setHint = (msg, color) => { hint.style.color = color; hint.textContent = msg; };
  if (!project_id) { setHint("Enter the Project ID first.", "#e5c07b"); return; }
  const key = _nebiusKeyPayload(null);
  if (!key) { setHint("Enter the key above first.", "#e5c07b"); return; }
  setHint("Looking up subnets…", "#e5c07b");
  try {
    const resp = await fetch("/api/training/nebius/discover/subnets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...key, project_id }),
    });
    const data = await resp.json().catch(() => null);
    if (!resp.ok) {
      setHint((data && data.detail) || `Lookup failed (${resp.status}). Enter the subnet ID manually.`, "#e06c75");
      return;
    }
    const subnets = data || [];
    if (subnets.length === 0) {
      select.style.display = "none";
      setHint("No subnets found in that project.", "#e5c07b");
    } else if (subnets.length === 1) {
      select.style.display = "none";
      input.value = subnets[0].id;
      setHint(`✓ Using ${subnets[0].name}`, "#98c379");
    } else {
      select.innerHTML = subnets
        .map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.name)} (${escapeHtml(s.id)})</option>`)
        .join("");
      select.style.display = "block";
      input.value = subnets[0].id;
      setHint(`${subnets.length} subnets — pick one`, "#98c379");
    }
  } catch (e) {
    setHint(`Lookup failed: ${e}. Enter the subnet ID manually.`, "#e06c75");
  }
}
function trainingPickSubnet() {
  const select = document.getElementById("nebius-conn-subnet-found");
  document.getElementById("nebius-conn-subnet").value = select.value;
}
async function trainingSaveNebiusConnection() {
  const statusEl = document.getElementById("nebius-conn-status");
  const key = _nebiusKeyPayload(statusEl);
  if (!key) return;
  const body = {
    ...key,
    project_id: document.getElementById("nebius-conn-project").value.trim(),
    subnet_id: document.getElementById("nebius-conn-subnet").value.trim(),
  };
  if (!body.project_id || !body.subnet_id) {
    statusEl.textContent = "Project ID and Subnet ID are required.";
    return;
  }
  statusEl.style.color = "#e5c07b";
  statusEl.textContent = "Saving…";
  try {
    const resp = await fetch("/api/training/nebius/connection", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      statusEl.style.color = "#e06c75";
      statusEl.textContent = err.detail || `Save failed (${resp.status}).`;
      return;
    }
    _nebiusConnStatus = await resp.json();
    document.getElementById("nebius-conn-key").value = "";
    trainingRefreshNebiusConnStatusLine();
    trainingCloseNebiusConnection();
    if (typeof showToast === "function") showToast("Nebius connection saved.", "success");
  } catch (e) {
    statusEl.style.color = "#e06c75";
    statusEl.textContent = `Save failed: ${e}`;
  }
}
async function trainingClearNebiusConnection() {
  if (!window.confirm("Remove the stored Nebius service-account key from this server?")) return;
  try {
    await fetch("/api/training/nebius/connection", { method: "DELETE" });
  } catch { /* ignore */ }
  await trainingFetchNebiusConnection();
  document.getElementById("nebius-conn-current").textContent = _nebiusConnSummary(_nebiusConnStatus);
  document.getElementById("nebius-conn-key").value = "";
  trainingRefreshNebiusConnStatusLine();
}
// The inline status line under the ephemeral Add-host help. It carries the
// connection's only management actions: "set it up" when absent, "disconnect"
// when present — so the dialog itself is just enter/replace-key (Cancel/Connect).
function trainingRefreshNebiusConnStatusLine() {
  const el = document.getElementById("add-host-nebius-conn-status");
  if (!el) return;
  const st = _nebiusConnStatus;
  if (st && st.configured) {
    const proj = st.project_id ? ` (${escapeHtml(st.project_id)})` : "";
    el.innerHTML =
      `<span style="color:#98c379;">✓ Connected${proj}</span> · ` +
      `<a href="#" onclick="trainingClearNebiusConnection(); return false;" style="color:var(--accent,#0e639c);">disconnect</a>`;
  } else {
    el.innerHTML =
      `<span style="color:#e5c07b;">✗ Nebius not connected</span> — ` +
      `<a href="#" onclick="trainingOpenNebiusConnection(); return false;" style="color:var(--accent,#0e639c);">set it up</a>`;
  }
}
let _trainingDatasets = [];
let _trainingPolicyCatalog = []; // populated by trainingLoadPolicies()
let _trainingPollTimer = null;

// View mode: which thing the main pane is showing.
//   "empty"  — nothing selected (default)
//   "form"   — start-training form open
//   "detail" — run detail for _trainingSelectedRunId
//
// Tracking mode explicitly fixes the "polling overwrites the start form"
// bug — refresh logic only redraws the view that matches the current mode.
let _trainingMode = "empty";
let _trainingSelectedRunId = null;

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  trainingInit();
});

async function trainingInit() {
  await trainingLoadHosts();
  await trainingRefreshRuns();
  trainingSchedulePoll();
}

function trainingSchedulePoll() {
  if (_trainingPollTimer) clearInterval(_trainingPollTimer);
  _trainingPollTimer = setInterval(trainingRefreshRuns, TRAINING_POLL_MS);
}

// ── Data loaders ──────────────────────────────────────────────────────────────

async function trainingLoadHosts() {
  try {
    const resp = await fetch("/api/training/hosts");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    _trainingHosts = await resp.json();
  } catch (e) {
    _trainingHosts = [];
    console.error("training: failed to load hosts", e);
  }
  trainingRenderHostsInfo();
}

// Fetch the policy catalog from the backend. Auto-discovered from
// PreTrainedConfig.get_known_choices() + HVLA's manual entry — see
// scripts/training/README.md § "Policies in the GUI". Cached for the
// lifetime of the page; refresh requires a reload (rare event).
async function trainingLoadPolicies() {
  if (_trainingPolicyCatalog.length > 0) return; // cached
  try {
    const resp = await fetch("/api/training/policies");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    _trainingPolicyCatalog = await resp.json();
  } catch (e) {
    _trainingPolicyCatalog = [];
    console.error("training: failed to load policy catalog", e);
  }
}

async function trainingLoadDatasets() {
  // Fan out: list sources → list each source's datasets → flatten.
  _trainingDatasets = [];
  try {
    const sources = await fetch("/api/datasets/sources").then((r) => r.json());
    const lists = await Promise.all(
      sources.map(async (s) => {
        try {
          const enc = encodeURIComponent(s.path);
          const ds = await fetch(`/api/datasets/sources/${enc}/datasets`).then((r) => r.json());
          return ds.map((d) => ({
            name: d.name,
            episodes: d.total_episodes,
            frames: d.total_frames,
            source: s.path,
          }));
        } catch {
          return [];
        }
      })
    );
    _trainingDatasets = lists.flat().sort((a, b) => a.name.localeCompare(b.name));
  } catch (e) {
    console.error("training: failed to load datasets", e);
  }
}

async function trainingRefreshRuns() {
  let runs = [];
  try {
    const resp = await fetch("/api/training/runs");
    if (resp.ok) runs = await resp.json();
  } catch (e) {
    console.error("training: failed to load runs", e);
  }
  trainingRenderRunsList(runs);
  // Only refresh the detail when we're currently SHOWING a detail. Don't
  // overwrite the start form with a re-render every 3 seconds.
  if (_trainingMode === "detail" && _trainingSelectedRunId) {
    await trainingRefreshDetail(_trainingSelectedRunId);
  }
}

// ── Sidebar render ────────────────────────────────────────────────────────────

function trainingRenderHostsInfo() {
  const el = document.getElementById("training-hosts-info");
  const btn = document.getElementById("training-start-btn");
  if (!el) return;
  if (_trainingHosts.length === 0) {
    el.innerHTML =
      '<div style="opacity:0.8;">No training hosts yet. Click + Host to add a remote SSH host, or attach a GPU on this machine.</div>';
    if (btn) {
      btn.disabled = true;
      btn.title = "No training hosts yet (workstation mode needs a GPU on this machine, or click + Host)";
    }
    return;
  }
  const rows = _trainingHosts.map((h) => {
    const removable = h.transport_kind !== "subprocess";  // anything but the local workstation
    const isEph = h.transport_kind === "ephemeral";
    const gpu = h.capabilities?.gpu_name || (isEph ? "cloud" : h.transport_kind === "ssh" ? "remote SSH" : "GPU");
    const vram = h.capabilities?.vram_mb ? ` · ${(h.capabilities.vram_mb / 1024).toFixed(1)} GB` : "";
    const tag = isEph ? ' <span style="opacity:0.6;">(ephemeral)</span>' : "";
    const removeBtn = removable
      ? ` <button class="btn-small secondary" style="margin-left:6px; padding:0 6px;" onclick="trainingDeleteHost('${cssEscape(h.id)}', '${cssEscape(h.display_name)}')" title="Remove this host">×</button>`
      : "";
    return `<div>${escapeHtml(h.display_name)} — ${escapeHtml(gpu)}${escapeHtml(vram)}${tag}${removeBtn}</div>`;
  });
  el.innerHTML = rows.join("");
  if (btn) {
    btn.disabled = false;
    btn.title = "Start a new training run";
  }
}

function trainingRenderRunsList(runs) {
  const el = document.getElementById("training-runs-list");
  if (!el) return;
  if (runs.length === 0) {
    el.innerHTML = '<div class="training-empty-hint">No runs yet. Click + to start.</div>';
    return;
  }
  el.innerHTML = "";
  for (const r of runs) {
    const row = document.createElement("div");
    row.className = "training-run-row";
    if (r.run_id === _trainingSelectedRunId && _trainingMode === "detail") {
      row.classList.add("selected");
    }
    row.innerHTML = `
      <div class="training-run-row-top">
        <span class="training-run-name">${escapeHtml(r.recipe_name)}</span>
        <span class="training-state-badge training-state-${r.state}">${r.state}</span>
      </div>
      <div class="training-run-row-sub">${escapeHtml(r.dataset_id)}</div>
    `;
    row.onclick = () => trainingSelectRun(r.run_id);
    // Right-click → minimal context menu. We override the browser default
    // (which would just show "Open / Save / Inspect…") with our own row
    // of actions. Currently the only action is duplicate; the menu is
    // structured so we can grow it (Delete / Resume) without touching
    // the row code.
    row.oncontextmenu = (ev) => {
      ev.preventDefault();
      trainingShowRunContextMenu(r, ev.clientX, ev.clientY);
    };
    el.appendChild(row);
  }
}

const TERMINAL_STATES = new Set(["completed", "failed", "aborted"]);

// One floating context-menu element shared across rows; created on first
// use, hidden by default, repositioned on each show. "Delete this run" is
// only rendered for terminal runs — active runs need a Stop first.
function trainingShowRunContextMenu(run, x, y) {
  let menu = document.getElementById("training-context-menu");
  if (!menu) {
    menu = document.createElement("div");
    menu.id = "training-context-menu";
    menu.className = "training-context-menu";
    document.body.appendChild(menu);
  }
  const isTerminal = TERMINAL_STATES.has(run.state);
  const deleteItem = isTerminal
    ? `
      <div class="training-context-item training-context-item-danger" data-action="delete">
        <span class="training-context-icon">🗑</span> Delete this run
      </div>
    `
    : "";
  menu.innerHTML = `
    <div class="training-context-item" data-action="duplicate">
      <span class="training-context-icon">⎘</span> Run with same config
    </div>
    ${deleteItem}
  `;
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  menu.style.display = "block";

  const close = () => {
    menu.style.display = "none";
    document.removeEventListener("click", close, true);
    document.removeEventListener("contextmenu", close, true);
  };
  menu.querySelector('[data-action="duplicate"]').onclick = (ev) => {
    ev.stopPropagation();
    close();
    trainingDuplicateRun(run.run_id);
  };
  const del = menu.querySelector('[data-action="delete"]');
  if (del) {
    del.onclick = (ev) => {
      ev.stopPropagation();
      close();
      trainingDeleteRun(run.run_id, run.recipe_name);
    };
  }
  // Close on next outside-click / right-click anywhere.
  setTimeout(() => {
    document.addEventListener("click", close, true);
    document.addEventListener("contextmenu", close, true);
  }, 0);
}

async function trainingDeleteRun(runId, label) {
  const ok = window.confirm(
    `Drop run "${label}" from training history? ` +
      `The trained model (if any) stays in the Models tab — only the run record disappears.`,
  );
  if (!ok) return;
  try {
    const resp = await fetch(`/api/training/runs/${runId}`, { method: "DELETE" });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const body = await resp.json();
    console.log(
      `training: dropped ${runId} from history ` +
        `(${body.metadata_bytes_freed} bytes metadata freed, kept_model=${body.kept_model})`,
    );
    // If the deleted run is the currently-selected detail, fall back to empty.
    if (_trainingSelectedRunId === runId) {
      trainingShowMain("empty");
      _trainingSelectedRunId = null;
    }
    trainingRefreshRuns();
  } catch (e) {
    alert(`Failed to delete run: ${e.message}`);
  }
}

async function trainingClearCompleted() {
  // Count terminal runs first so the confirm dialog has a real number.
  let runs = [];
  try {
    runs = await fetch("/api/training/runs").then((r) => r.json());
  } catch (_) {
    /* fall through with empty list */
  }
  const terminal = runs.filter((r) => TERMINAL_STATES.has(r.state));
  if (terminal.length === 0) {
    alert("No completed runs to clear.");
    return;
  }
  const ok = window.confirm(
    `Drop ${terminal.length} completed/failed/aborted run(s) from training history? ` +
      `Trained models (if any) stay in the Models tab — only the run records disappear.`,
  );
  if (!ok) return;
  try {
    const resp = await fetch("/api/training/runs/clear", { method: "POST" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const body = await resp.json();
    console.log(
      `training: cleared ${body.deleted.length} runs ` +
        `(${body.metadata_bytes_freed} bytes metadata freed, ${body.models_kept} models kept)`,
    );
    // If the selected run was in the cleared set, fall back to empty.
    if (_trainingSelectedRunId && body.deleted.includes(_trainingSelectedRunId)) {
      trainingShowMain("empty");
      _trainingSelectedRunId = null;
    }
    trainingRefreshRuns();
  } catch (e) {
    alert(`Failed to clear runs: ${e.message}`);
  }
}

// ── Mode switches ─────────────────────────────────────────────────────────────

function trainingShowMain(mode) {
  // Hide existing model views; show the training-detail container for any
  // mode other than "empty".
  document.getElementById("model-empty").style.display = mode === "empty" ? "" : "none";
  document.getElementById("model-detail").style.display = "none";
  document.getElementById("training-detail").style.display = mode === "empty" ? "none" : "block";
  _trainingMode = mode;
}

function trainingSelectRun(runId) {
  _trainingSelectedRunId = runId;
  trainingShowMain("detail");
  trainingRefreshDetail(runId);
  trainingRefreshRuns(); // re-render sidebar selection highlight
}

// Called by the Model tab when it shows model-* content. Hides the training
// pane and resets internal state so the poll loop won't re-render on top of
// the model view. Symmetric to trainingShowMain() hiding model containers
// when the user enters training mode.
function trainingLeaveView() {
  const td = document.getElementById("training-detail");
  if (td) td.style.display = "none";
  _trainingMode = "empty";
  _trainingSelectedRunId = null;
  trainingRefreshRuns(); // clear sidebar highlight
}

// ── Detail pane ───────────────────────────────────────────────────────────────

async function trainingRefreshDetail(runId) {
  const el = document.getElementById("training-detail");
  if (!el || _trainingMode !== "detail") return;
  try {
    const resp = await fetch(`/api/training/runs/${runId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const snap = await resp.json();
    // Preserve user scroll across the 2s full-pane re-render. Two scrolls
    // to track: the pane itself (form/run-detail height) and the log-tail
    // <pre> inside it. For the log, "stick to bottom" if the user is
    // already near the bottom (auto-follow), otherwise restore exact
    // position. SCROLL_STICKY_PX = 32 absorbs sub-pixel rounding from
    // browser zoom + lets a near-bottom user re-anchor cleanly.
    const SCROLL_STICKY_PX = 32;
    const paneScroll = el.scrollTop;
    const prevLog = el.querySelector(".training-log");
    let logScroll = 0;
    let logStuckToBottom = false;
    if (prevLog) {
      logScroll = prevLog.scrollTop;
      logStuckToBottom =
        prevLog.scrollHeight - prevLog.scrollTop - prevLog.clientHeight < SCROLL_STICKY_PX;
    }
    el.innerHTML = trainingRenderDetailHtml(snap);
    trainingDrawDetailCharts(snap); // canvas charts need the DOM in place
    el.scrollTop = paneScroll;
    const newLog = el.querySelector(".training-log");
    if (newLog) {
      newLog.scrollTop = logStuckToBottom ? newLog.scrollHeight : logScroll;
    }
    const stopBtn = document.getElementById(`training-stop-${runId}`);
    if (stopBtn) stopBtn.onclick = () => trainingStopRun(runId);
    const cloneBtn = document.getElementById(`training-clone-${runId}`);
    if (cloneBtn) cloneBtn.onclick = () => trainingDuplicateRun(runId);
  } catch (e) {
    el.innerHTML = `<div class="training-error">Failed to load run: ${escapeHtml(e.message)}</div>`;
  }
}

// Format a metric value compactly: tiny/huge → scientific (1.0e-5), else
// up to 4 significant decimals trimmed of trailing zeros.
function trainingFmtMetric(v) {
  const n = Number(v);
  if (!isFinite(n)) return "—";
  if (n !== 0 && (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e5)) return n.toExponential(1);
  return parseFloat(n.toFixed(4)).toString();
}

// Seconds → "1h 5m" / "12m 30s" / "45s". Used for ETA + elapsed.
function trainingFmtDuration(s) {
  s = Math.max(0, Math.round(Number(s) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

// First Weights & Biases run URL in the log tail, if the run uses wandb.
function trainingWandbUrl(text) {
  if (!text) return null;
  const m = text.match(/https:\/\/wandb\.ai\/[^\s/]+\/[^\s/]+\/runs\/[A-Za-z0-9]+/);
  return m ? m[0] : null;
}

// Curated default metric charts (loss, lr). These reuse the SAME canvas
// primitive (`drawChart` in charts.js) as the RLT + performance panels so the
// app's charts look and behave consistently — including hover/crosshair. The
// canvases are drawn post-render by trainingDrawDetailCharts(). grad_norm +
// other auto-captured keys live in the stat tiles / a future metric picker.
const TRAINING_CHART_KEYS = [
  { key: "loss", label: "Loss", color: "#34d399", logY: true }, // spans orders of magnitude
  { key: "lr", label: "Learning rate", color: "#fb923c" },
];

function trainingMetricsCardHtml(series, isActive) {
  if (!series.length) {
    return isActive
      ? '<section class="training-card"><div class="training-empty-hint">Metrics will appear once training logs its first step…</div></section>'
      : "";
  }
  const card = (c) =>
    `<div class="training-chart">
       <div class="training-chart-title">${c.label}</div>
       <canvas id="training-chart-${c.key}" class="training-chart-canvas"></canvas>
     </div>`;
  return `
    <section class="training-card">
      <h3 class="training-card-heading">Metrics</h3>
      <div class="training-charts">${TRAINING_CHART_KEYS.map(card).join("")}</div>
    </section>`;
}

// Draw the metric canvases after the detail HTML is in the DOM (canvas needs
// layout for its getBoundingClientRect). Uses the shared drawChart primitive
// (charts.js); the 'training' sync group gives loss + lr a shared crosshair.
function trainingDrawDetailCharts(snap) {
  if (typeof drawChart !== "function") return; // provided by charts.js
  const series = (snap.metrics || []).filter((m) => typeof m.step === "number");
  const latestStep = series.length ? series[series.length - 1].step : 0;
  for (const c of TRAINING_CHART_KEYS) {
    const data = series.map((m) => m[c.key]).filter((v) => typeof v === "number" && isFinite(v));
    if (data.length) {
      drawChart(`training-chart-${c.key}`, {
        series: [{ data, color: c.color, label: c.label }],
        syncGroup: "training",
        latestStep,
        logY: !!c.logY,
      });
    }
  }
}

function trainingRenderDetailHtml(snap) {
  const r = snap.run;
  const progress = snap.progress || {};
  const checkpoints = snap.checkpoints || [];
  const events = snap.events || [];
  const isActive = !["completed", "failed", "aborted"].includes(r.state);

  // Position comes from progress.json (parsed from the tqdm bar by the
  // orchestrator). The training-signal series (loss/lr/grdn) comes from
  // metrics.jsonl. progress.total_steps/eta_seconds are the real-run fields;
  // num_steps/loss are the legacy fake-runner schema — read both so the UI
  // works during the transition (and degrades to checkpoint-step when no
  // progress exists at all).
  //
  // Checkpoint step uses max-by-step, not last entry: the scanner sorts dirs
  // alphabetically, so HVLA's ``checkpoint-10`` can land before ``checkpoint-5``.
  const metricsSeries = (snap.metrics || []).filter((m) => typeof m.step === "number");
  const latest = metricsSeries.length ? metricsSeries[metricsSeries.length - 1] : {};
  const lastCkptStep = checkpoints.length ? Math.max(...checkpoints.map((c) => c.step)) : 0;
  const step = progress.step ?? latest.step ?? lastCkptStep;
  const total = progress.total_steps ?? progress.num_steps ?? r.args?.num_steps ?? r.args?.steps ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((step / total) * 100)) : 0;
  // loss/lr/grad: prefer the parsed metric series; fall back to the fake
  // runner's progress.loss so legacy/test runs still show a value.
  const lossVal = latest.loss ?? progress.loss;
  const loss = lossVal != null ? trainingFmtMetric(lossVal) : "—";
  const lr = latest.lr != null ? trainingFmtMetric(latest.lr) : "—";
  const grdn = latest.grdn != null ? trainingFmtMetric(latest.grdn) : "—";
  const eta = progress.eta_seconds != null ? trainingFmtDuration(progress.eta_seconds) : "—";
  // Running but no step parsed yet → tqdm hasn't printed its first bar.
  const warming = isActive && step === 0;
  const wandbUrl = trainingWandbUrl(snap.stderr_tail);
  // For terminal runs, freeze the elapsed clock at finished_at instead of
  // ticking forever against Date.now().
  const elapsedEnd = r.finished_at ?? Date.now() / 1000;
  const elapsedSec = r.started_at ? Math.max(0, Math.round(elapsedEnd - r.started_at)) : 0;

  const stopBtn = isActive
    ? `<button class="btn-small danger" id="training-stop-${r.run_id}">Stop</button>`
    : "";
  // "Run with same config" is always available — works for in-progress
  // runs (peek at the config), terminal runs (clone-and-adjust), and
  // failed runs (fix-and-retry).
  const cloneBtn = `<button class="btn-small secondary" id="training-clone-${r.run_id}" title="Open the start form pre-filled with this run's settings">Run with same config</button>`;

  // Image-prep status banner: when state=PENDING (background prep thread is
  // running), surface the most recent image-* event so the user sees what's
  // happening rather than a blank pending state. After RUNNING/terminal,
  // collapse to a one-liner ("Image: cache hit" / "Image: pulled 14.6 GB
  // in 13.5 min") that explains the silent setup time.
  const imageBanner = trainingImageStatusHtml(r, events);

  return `
    <div class="training-detail-pane">
      <header class="training-detail-header">
        <div>
          <h2 class="training-detail-title">${escapeHtml(r.recipe_name)}</h2>
          <div class="training-detail-meta">
            <span>${escapeHtml(r.dataset_id)}</span>
            <span class="sep">·</span>
            <span>host ${escapeHtml(r.host_id)}</span>
            <span class="sep">·</span>
            <span class="training-mono">${escapeHtml(r.run_id)}</span>
          </div>
        </div>
        <div class="training-detail-actions">
          <span class="training-state-badge training-state-${r.state}">${r.state}</span>
          ${cloneBtn}
          ${stopBtn}
        </div>
      </header>

      ${imageBanner}

      <section class="training-card">
        <div class="training-stats-row">
          <div class="training-stat"><div class="training-stat-label">Step</div><div class="training-stat-value">${step}${total ? " / " + total : ""}</div></div>
          <div class="training-stat"><div class="training-stat-label">Loss</div><div class="training-stat-value">${loss}</div></div>
          <div class="training-stat"><div class="training-stat-label">LR</div><div class="training-stat-value">${lr}</div></div>
          <div class="training-stat"><div class="training-stat-label">Grad norm</div><div class="training-stat-value">${grdn}</div></div>
          <div class="training-stat"><div class="training-stat-label">ETA</div><div class="training-stat-value">${eta}</div></div>
          <div class="training-stat"><div class="training-stat-label">Elapsed</div><div class="training-stat-value">${trainingFmtDuration(elapsedSec)}</div></div>
          <div class="training-stat"><div class="training-stat-label">Checkpoints</div><div class="training-stat-value">${checkpoints.length}</div></div>
        </div>
        <div class="training-progress-bar">
          <div class="training-progress-fill" style="width: ${pct}%"></div>
          <span class="training-progress-label">${warming ? "warming up…" : pct + "%"}</span>
        </div>
        ${
          wandbUrl
            ? `<div class="training-field-hint">📊 <a href="${escapeHtml(wandbUrl)}" target="_blank" rel="noopener" style="color:var(--accent,#4fc3f7);">View run in Weights &amp; Biases ↗</a></div>`
            : ""
        }
      </section>

      ${trainingMetricsCardHtml(metricsSeries, isActive)}

      <section class="training-card">
        <h3 class="training-card-heading">Checkpoints</h3>
        ${
          checkpoints.length === 0
            ? '<div class="training-empty-hint">None yet</div>'
            : `<ul class="training-checkpoint-list">${checkpoints
                .map(
                  (c) => `<li>
                <span class="training-mono">step ${c.step}</span>
                <span class="training-mono training-muted">${escapeHtml(c.path)}</span>
                <span class="training-mono training-muted">${c.sha256.slice(0, 12)}…</span>
              </li>`
                )
                .join("")}</ul>`
        }
      </section>

      <section class="training-card">
        <h3 class="training-card-heading">Log tail</h3>
        <pre class="training-log">${escapeHtml(snap.stderr_tail || "(no output yet)")}</pre>
      </section>

      ${trainingConfigCardHtml(r)}

      ${r.error ? `<div class="training-error">Error: ${escapeHtml(r.error)}</div>` : ""}
    </div>
  `;
}

// Configuration card: the same args dict that started this run, surfaced
// so the user can read it back and decide whether to clone. Filters out
// recipe-builder meta keys (the leading-`__` ones) — they don't change
// behaviour the user controls.
function trainingConfigCardHtml(r) {
  const args = r.args || {};
  const keys = Object.keys(args)
    .filter((k) => !k.startsWith("__"))
    .sort();
  const rows = keys.map((k) => {
    const v = args[k];
    const valStr = typeof v === "object" ? JSON.stringify(v) : String(v);
    return `<tr><td class="training-mono">${escapeHtml(k)}</td><td class="training-mono">${escapeHtml(valStr)}</td></tr>`;
  });
  // Recipe + image markers belong with the config but get their own row,
  // labeled clearly. They're what determined which trainer ran + which
  // image was used.
  const recipeMarker = args["__recipe__"] || "lerobot-train";
  const imageMarker = args["__image__"] || "(default)";
  return `
    <section class="training-card">
      <details class="training-section" open>
        <summary class="training-section-summary">Configuration</summary>
        <table class="training-args-table">
          <tr><th>Recipe</th><td class="training-mono">${escapeHtml(recipeMarker)}</td></tr>
          <tr><th>Image</th><td class="training-mono">${escapeHtml(imageMarker)}</td></tr>
          ${rows.join("")}
        </table>
        <div class="training-field-hint">Click <strong>Run with same config</strong> above to open the start form pre-filled with these values.</div>
      </details>
    </section>
  `;
}

// ── Start form ────────────────────────────────────────────────────────────────

async function trainingShowStartForm(prefill) {
  // Button is disabled until hosts load, so this guard mostly catches
  // programmatic invocations. Keep it cheap and silent — no alert.
  if (_trainingHosts.length === 0) return;
  // Clear selected run so polling doesn't try to refresh detail over the form.
  _trainingSelectedRunId = null;
  trainingShowMain("form");
  // Load datasets + policy catalog lazily on first form-open. Both are
  // network-fetched and cheap to cache for the page lifetime.
  if (_trainingDatasets.length === 0 || _trainingPolicyCatalog.length === 0) {
    document.getElementById("training-detail").innerHTML =
      '<div class="training-detail-pane"><div class="training-empty-hint">Loading datasets + policy catalog…</div></div>';
    await Promise.all([trainingLoadDatasets(), trainingLoadPolicies()]);
  }
  if (_trainingMode === "form") trainingRenderStartForm(prefill); // user might have nav'd away
  trainingRefreshRuns(); // unselect any previously-highlighted row
}

// Duplicate an existing run: open the start form pre-filled with that
// run's policy + dataset + hyperparameters. Used by the per-row context
// menu and the "Run with same config" button in the detail view.
async function trainingDuplicateRun(runId) {
  let snap;
  try {
    const resp = await fetch(`/api/training/runs/${runId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    snap = await resp.json();
  } catch (e) {
    alert(`Failed to load run config: ${e.message}`);
    return;
  }
  const r = snap.run;
  trainingShowStartForm({
    dataset_id: r.dataset_id,
    args: r.args || {},
    recipe_name: r.recipe_name ? `${r.recipe_name} (copy)` : "",
  });
}

// ── Image-prep status banner ──────────────────────────────────────────────────
//
// Reads the events.jsonl-derived `events` list off the snapshot and renders
// a one-line status describing the most recent image-* event. The
// orchestrator emits exactly one of these flows per run:
//
//   1. cache hit:       image_cache_hit
//   2. successful pull: image_pull_started → image_pulled (with duration_s + size_bytes)
//   3. failed pull:     image_pull_started → image_pull_failed (with error tail)
//
// Why surface this: a first-time pull on a fresh host took 13m 34s on the
// reference workstation. Without this banner the run sits at PENDING with
// nothing on screen, which is indistinguishable from a hung backend.
function trainingImageStatusHtml(run, events) {
  const imageEvents = events.filter((e) => typeof e.type === "string" && e.type.startsWith("image"));
  if (imageEvents.length === 0) {
    if (run.state === "pending") {
      return `<div class="training-image-banner pending">Preparing image…</div>`;
    }
    return ""; // running / terminal with no image events — fake recipe path
  }
  const last = imageEvents[imageEvents.length - 1];
  switch (last.type) {
    case "image_cache_hit":
      return `<div class="training-image-banner ok">Image: cache hit · <span class="training-mono">${escapeHtml(last.image)}</span></div>`;
    case "image_pull_started": {
      const sinceMs = last.ts ? Date.now() - last.ts * 1000 : 0;
      const sinceLabel = sinceMs > 0 ? ` · pulling for ${formatDuration(sinceMs / 1000)}` : "";
      return `<div class="training-image-banner pulling">Pulling image · <span class="training-mono">${escapeHtml(last.image)}</span>${sinceLabel}</div>`;
    }
    case "image_pulled": {
      const dur = last.duration_s != null ? formatDuration(last.duration_s) : "?";
      const size = last.size_bytes != null ? ` · ${formatBytes(last.size_bytes)}` : "";
      return `<div class="training-image-banner ok">Image: pulled in ${dur}${size} · <span class="training-mono">${escapeHtml(last.image)}</span></div>`;
    }
    case "image_pull_failed":
      return `<div class="training-image-banner failed">Image pull failed: <span class="training-mono">${escapeHtml(last.error || "(no error tail)")}</span></div>`;
    default:
      return "";
  }
}

function formatDuration(seconds) {
  const s = Math.max(0, seconds);
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  return `${m}m ${r}s`;
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Policy catalog is fetched at form-load from GET /api/training/policies.
// The backend auto-discovers every PreTrainedConfig subclass (via
// `@PreTrainedConfig.register_subclass(...)`) plus manually-registered
// non-draccus recipes (HVLA). Adding a new lerobot-train-registered
// policy upstream surfaces in this picker with NO frontend code edit.
// See scripts/training/README.md § "Policies in the GUI" for the data
// shape + the field-renderable type matrix.
//
// One adapter helper: each catalog entry is
// ``{type_name, label, recipe, arg_key_prefix, fields: [{name, label,
// type, default, choices?, description?}]}``. The frontend prepends
// ``arg_key_prefix`` (``"policy."`` for draccus recipes, ``""`` for HVLA)
// to each field's ``name`` when building the args dict that POST
// /api/training/runs receives.

function trainingPolicyEntry(typeName) {
  return _trainingPolicyCatalog.find((p) => p.type_name === typeName);
}

// Field-key the form input uses to back a policy field. The backend
// returns the bare ``name`` (e.g. ``chunk_size``); the form's input name
// has to include the args-dict key (``policy.chunk_size`` for draccus,
// ``chunk_size`` for HVLA) so trainingSubmitStart can read it back via
// FormData with the right key. Returning the prefixed key here keeps
// both sides aligned.
function trainingFormKey(policyEntry, field) {
  return (policyEntry?.arg_key_prefix || "") + field.name;
}

// Common training fields shared across all policies. Copied from
// src/lerobot/configs/train.py defaults.
const TRAINING_FIELDS = [
  { key: "steps", label: "Total training steps", type: "int", default: 1000 },
  { key: "batch_size", label: "Batch size", type: "int", default: 8 },
  { key: "save_freq", label: "Save every N steps", type: "int", default: 500 },
];

function trainingRenderStartForm(prefill) {
  const el = document.getElementById("training-detail");
  if (!el || _trainingMode !== "form") return;

  const hostOptions = _trainingHosts
    .map((h) => `<option value="${escapeHtml(h.id)}">${escapeHtml(h.display_name)}</option>`)
    .join("");

  const datasetOptions =
    _trainingDatasets.length === 0
      ? `<option value="" disabled>No datasets found — add a source in the Data tab first</option>`
      : _trainingDatasets
          .map(
            (d) =>
              `<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)} (${d.episodes} ep · ${d.frames} fr)</option>`
          )
          .join("");

  const policyOptions =
    _trainingPolicyCatalog.length === 0
      ? `<option value="" disabled>Loading policies…</option>`
      : _trainingPolicyCatalog
          .map((p) => `<option value="${escapeHtml(p.type_name)}">${escapeHtml(p.label)}</option>`)
          .join("");

  el.innerHTML = `
    <div class="training-detail-pane">
      <header class="training-detail-header">
        <h2 class="training-detail-title">Start a training run</h2>
      </header>

      <form id="training-start-form" class="training-form" onsubmit="trainingSubmitStart(event); return false;">
        <label class="training-field">
          <span class="training-field-label">Run name</span>
          <input type="text" name="recipe_name" value="" placeholder="(optional — defaults to <policy>-<dataset_basename>)" />
          <span class="training-field-hint">Names the trained model. It appears under this name in the run list + Models tab. The output location is managed automatically (under the GUI's runs dir) so the Models tab can find it — it isn't a free path. Leave blank for auto.</span>
        </label>

        <label class="training-field">
          <span class="training-field-label">Host</span>
          <select name="host_id" required>${hostOptions}</select>
        </label>

        <label class="training-field">
          <span class="training-field-label">Policy</span>
          <select name="policy_type" required onchange="trainingRenderPolicyFields(this.value)">
            ${policyOptions}
          </select>
          <span class="training-field-hint">Each policy renders its own hyperparameter form below. Defaults come from upstream lerobot config classes.</span>
        </label>

        <label class="training-field">
          <span class="training-field-label">Dataset</span>
          <select name="dataset_id" required ${_trainingDatasets.length === 0 ? "disabled" : ""}>
            ${datasetOptions}
          </select>
          <span class="training-field-hint">Datasets are discovered from sources configured in the Data tab.</span>
        </label>

        <details class="training-section" open>
          <summary class="training-section-summary">Policy hyperparameters</summary>
          <div id="training-policy-fields"><!-- populated by trainingRenderPolicyFields --></div>
        </details>

        <details class="training-section" open>
          <summary class="training-section-summary">Training</summary>
          <div class="training-field-row">
            ${TRAINING_FIELDS.map((f) => fieldHtml(f)).join("")}
          </div>
        </details>

        <div class="training-form-actions">
          <button type="submit" class="btn-small">Start training</button>
          <button type="button" class="btn-small secondary" onclick="trainingCancelForm()">Cancel</button>
        </div>
        <div id="training-start-error" class="training-error" style="display:none;"></div>
      </form>
    </div>
  `;
  // Render the policy-specific fields. If we have a prefill, derive the
  // policy from its args; otherwise fall back to the first catalog entry.
  const initialPolicy =
    trainingPolicyFromArgs(prefill?.args) || _trainingPolicyCatalog[0]?.type_name || "";
  trainingRenderPolicyFields(initialPolicy);
  if (prefill) trainingApplyPrefill(prefill, initialPolicy);
}

// Map a Run.args dict back to its policy_type key. Inverse of
// trainingSubmitStart's args-construction:
//   - lerobot-train recipes carry `policy.type=<key>`
//   - non-draccus recipes (e.g. HVLA) carry `__recipe__=<marker>` instead
//     of `policy.type`. The catalog entry's `recipe` field is the marker
//     to match.
function trainingPolicyFromArgs(args) {
  if (!args) return null;
  if (typeof args["policy.type"] === "string") return args["policy.type"];
  const recipeMarker = args["__recipe__"];
  if (typeof recipeMarker === "string") {
    const entry = _trainingPolicyCatalog.find((p) => p.recipe === recipeMarker);
    if (entry) return entry.type_name;
  }
  return null;
}

// Populate the form from a previous run's snapshot. Called from
// trainingDuplicateRun → trainingShowStartForm → trainingRenderStartForm.
function trainingApplyPrefill(prefill, policyType) {
  const form = document.getElementById("training-start-form");
  if (!form) return;
  const policy = trainingPolicyEntry(policyType);

  // Dataset dropdown — only set if the option exists (dataset may have
  // been removed since the previous run; let the user notice).
  if (prefill.dataset_id) {
    const dsSel = form.querySelector("select[name=dataset_id]");
    if (dsSel && Array.from(dsSel.options).some((o) => o.value === prefill.dataset_id)) {
      dsSel.value = prefill.dataset_id;
    }
  }

  // Policy dropdown (the policy-fields are already rendered for
  // `policyType` by the caller; just set the select to match)
  const polSel = form.querySelector("select[name=policy_type]");
  if (polSel && policy) polSel.value = policyType;

  // Run label (kept verbatim — "(copy)" suffix added by trainingDuplicateRun)
  if (prefill.recipe_name) {
    const labelInput = form.querySelector("input[name=recipe_name]");
    if (labelInput) labelInput.value = prefill.recipe_name;
  }

  // Fill each field by its FORM KEY (= arg_key_prefix + field.name).
  // Catalog fields use bare ``name``; shared TRAINING_FIELDS use ``key``.
  // The input's HTML name attribute is the form key in both cases.
  const args = prefill.args || {};
  const catalogFields = (policy?.fields || []).map((f) => ({
    key: trainingFormKey(policy, f),
    type: f.type,
  }));
  const trainingFields = TRAINING_FIELDS.map((f) => ({ key: f.key, type: f.type }));
  for (const f of [...trainingFields, ...catalogFields]) {
    if (!(f.key in args)) continue;
    const input = form.querySelector(`[name="${cssEscape(f.key)}"]`);
    if (!input) continue;
    if (f.type === "bool") {
      input.checked = !!args[f.key];
    } else {
      input.value = String(args[f.key]);
    }
  }
}

// Minimal CSS.escape() shim — names contain dots (e.g. `policy.chunk_size`)
// which `querySelector` would otherwise interpret as descendant selectors.
function cssEscape(s) {
  if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
  return String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => "\\" + c);
}

function trainingRenderPolicyFields(policyType) {
  const container = document.getElementById("training-policy-fields");
  if (!container) return;
  const policy = trainingPolicyEntry(policyType);
  if (!policy) {
    container.innerHTML = '<div class="training-empty-hint">Unknown policy</div>';
    return;
  }
  // Each backend field has bare ``name`` (e.g. ``chunk_size``). The form
  // input's HTML name attribute must be the args-dict key, so we prepend
  // ``arg_key_prefix`` here.
  const fields = policy.fields.map((f) => ({
    ...f,
    key: trainingFormKey(policy, f),
  }));
  container.innerHTML = `<div class="training-field-row">${fields.map(fieldHtml).join("")}</div>`;
}

function fieldHtml(f) {
  const id = `training-arg-${f.key.replace(/\./g, "-")}`;
  const labelText = escapeHtml(f.label);
  const desc = f.description ? `<span class="training-field-hint">${escapeHtml(f.description)}</span>` : "";
  if (f.type === "bool") {
    return `
      <label class="training-field training-field-bool">
        <span class="training-field-label">${labelText}</span>
        <input id="${id}" type="checkbox" name="${escapeHtml(f.key)}" ${f.default ? "checked" : ""} />
        ${desc}
      </label>
    `;
  }
  if (f.type === "select" && Array.isArray(f.choices)) {
    const opts = f.choices
      .map((c) => `<option value="${escapeHtml(c)}"${String(f.default) === c ? " selected" : ""}>${escapeHtml(c)}</option>`)
      .join("");
    return `
      <label class="training-field">
        <span class="training-field-label">${labelText}</span>
        <select id="${id}" name="${escapeHtml(f.key)}">${opts}</select>
        ${desc}
      </label>
    `;
  }
  // int/float/string — use a number input for numerics so step controls show.
  let inputAttrs;
  if (f.type === "int") inputAttrs = 'type="number" step="1"';
  else if (f.type === "float") inputAttrs = 'type="number" step="any"';
  else inputAttrs = 'type="text"';
  const defaultStr = f.default === null || f.default === undefined ? "" : escapeHtml(String(f.default));
  return `
    <label class="training-field">
      <span class="training-field-label">${labelText}</span>
      <input id="${id}" ${inputAttrs} name="${escapeHtml(f.key)}" value="${defaultStr}" />
      ${desc}
    </label>
  `;
}

function trainingCancelForm() {
  trainingShowMain("empty");
  trainingRefreshRuns();
}

async function trainingSubmitStart(ev) {
  ev?.preventDefault?.();
  const form = document.getElementById("training-start-form");
  if (!form) return;
  const fd = new FormData(form);
  const idempotencyKey = `${Date.now()}-${Math.random().toString(36).slice(2)}`;

  const hostId = fd.get("host_id");
  const datasetId = fd.get("dataset_id");
  const policyType = fd.get("policy_type");

  // Build the args dict the recipe builder expects. Shape depends on
  // the catalog entry:
  //   - draccus recipes (recipe = null): dotted keys via the
  //     ``arg_key_prefix`` ("policy."), policy.type carries the chosen
  //     policy class, dataset.repo_id holds the dataset.
  //   - non-draccus recipes (HVLA): the ``__recipe__`` marker routes to
  //     the matching backend builder, ``arg_key_prefix`` is empty so
  //     keys go in bare, and the dataset key is ``dataset_repo_id``
  //     (HVLA's argparse name).
  const policyEntry = trainingPolicyEntry(policyType);
  const recipe = policyEntry?.recipe || null;
  const args = recipe
    ? { __recipe__: recipe, dataset_repo_id: datasetId }
    : { "policy.type": policyType, "dataset.repo_id": datasetId };

  // Policy-specific fields (from the catalog). Each backend field has
  // bare ``name``; the form input's HTML name is the prefixed form key.
  for (const f of policyEntry?.fields || []) {
    const formKey = trainingFormKey(policyEntry, f);
    const v = formValue(fd, form, { ...f, key: formKey });
    if (v !== undefined) args[formKey] = v;
  }
  // Common training fields (snake_case keys — match HVLA flag names
  // verbatim; lerobot-train accepts them as top-level dataclass fields).
  for (const f of TRAINING_FIELDS) {
    const v = formValue(fd, form, f);
    if (v !== undefined) args[f.key] = v;
  }

  // Auto-generate a label if user didn't provide one
  let recipeName = (fd.get("recipe_name") || "").trim();
  if (!recipeName) {
    const dsBasename = datasetId.split("/").slice(-1)[0];
    recipeName = `${policyType}-${dsBasename}`;
  }

  const body = {
    host_id: hostId,
    recipe_name: recipeName,
    dataset_id: datasetId,
    args,
    idempotency_key: idempotencyKey,
  };
  // Ephemeral hosts spawn a vendor VM via the server-held Nebius connection.
  // Warn early (before submit) if it isn't configured, so the user fixes it
  // here rather than hitting a spawn failure mid-run.
  const startHost = _trainingHosts.find((h) => h.id === hostId);
  if (startHost && startHost.transport_kind === "ephemeral") {
    const st = await trainingFetchNebiusConnection();
    if (!st || !st.configured) {
      trainingOpenNebiusConnection();
      return;
    }
  }
  const errEl = document.getElementById("training-start-error");
  errEl.style.display = "none";
  const submitBtn = form.querySelector("button[type=submit]");
  if (submitBtn) submitBtn.disabled = true;
  try {
    const resp = await fetch("/api/training/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const run = await resp.json();
    // Switch to detail view of the new run
    trainingSelectRun(run.run_id);
  } catch (e) {
    errEl.style.display = "block";
    errEl.textContent = e.message || String(e);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

// ── Stop ──────────────────────────────────────────────────────────────────────

async function trainingStopRun(runId) {
  if (!confirm("Stop this training run?")) return;
  try {
    const resp = await fetch(`/api/training/runs/${runId}/stop`, { method: "POST" });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    await trainingRefreshDetail(runId);
    await trainingRefreshRuns();
  } catch (e) {
    alert(`Stop failed: ${e.message || e}`);
  }
}

// ── Utilities ────────────────────────────────────────────────────────────────

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formValue(fd, form, field) {
  if (field.type === "bool") {
    // Checkboxes only appear in FormData when checked; explicitly read the
    // input element to handle the unchecked case as `false`.
    const el = form.querySelector(`input[name="${field.key}"]`);
    return el ? !!el.checked : false;
  }
  const raw = fd.get(field.key);
  if (raw == null || raw === "") return undefined;
  if (field.type === "int") {
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : undefined;
  }
  if (field.type === "float") {
    const n = parseFloat(raw);
    return Number.isFinite(n) ? n : undefined;
  }
  return raw;
}

// Expose for the inline onclick handlers
window.trainingInit = trainingInit;
window.trainingLoadHosts = trainingLoadHosts;
window.trainingShowStartForm = trainingShowStartForm;
window.trainingCancelForm = trainingCancelForm;
window.trainingLeaveView = trainingLeaveView;
window.trainingSubmitStart = trainingSubmitStart;
window.trainingStopRun = trainingStopRun;
window.trainingOpenNebiusConnection = trainingOpenNebiusConnection;
window.trainingCloseNebiusConnection = trainingCloseNebiusConnection;
window.trainingSaveNebiusConnection = trainingSaveNebiusConnection;
window.trainingClearNebiusConnection = trainingClearNebiusConnection;
window.trainingRenderPolicyFields = trainingRenderPolicyFields;
window.trainingDuplicateRun = trainingDuplicateRun;
window.trainingDeleteRun = trainingDeleteRun;
window.trainingClearCompleted = trainingClearCompleted;
