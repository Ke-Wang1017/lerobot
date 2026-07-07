// Copyright 2026 The HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// "Add SSH host" dialog — POSTs a HostProfile to /api/training/hosts.
//
// State machine:
//   (closed)
//     ↓ "+ Host"
//   open, empty, save disabled
//     ↓ user fills Name + Host, clicks "Test"
//   probing (test button → "Testing…", disabled)
//     ↓ probe returns
//   probed (4-row checklist rendered, save enabled iff probe.ok)
//     ↓ user clicks Save
//   saving → success → close + refresh sidebar
//
// Saving is only allowed after a green probe — same posture as VS Code
// Remote-SSH's "verified" badge. Editing any field after a probe clears
// the result and re-disables Save (otherwise the user could pass a probe
// for host A and then save host B).

let _addHostLastProbeOk = false;
// Abort any in-flight probe on close/reopen — otherwise a stale response
// landing after the dialog was reset could re-enable Save for a host the
// user has since edited.
let _addHostProbeAbort = null;

function _abortPendingProbe() {
    if (_addHostProbeAbort) {
        _addHostProbeAbort.abort();
        _addHostProbeAbort = null;
    }
}

function openAddHostDialog() {
    const overlay = document.getElementById('add-host-overlay');
    if (!overlay) return;
    _abortPendingProbe();
    document.getElementById('add-host-name').value = '';
    document.getElementById('add-host-host').value = '';
    document.getElementById('add-host-probe-results').innerHTML = '';
    document.getElementById('add-host-status').textContent = '';
    document.getElementById('add-host-test-btn').textContent = 'Test';
    document.getElementById('add-host-type').value = 'ssh';
    _addHostLastProbeOk = false;
    addHostTypeChanged();  // sets per-type field visibility + button state
    overlay.style.display = 'flex';
    document.getElementById('add-host-name').focus();
}

function _addHostType() {
    const el = document.getElementById('add-host-type');
    return el ? el.value : 'ssh';
}

function _addHostNameValid() {
    const name = (document.getElementById('add-host-name').value || '').trim();
    return /^[a-zA-Z0-9_-]+$/.test(name);
}

// Save is enabled only with a valid Name AND (ephemeral, or a green SSH probe).
// Surfaces an inline hint when the Name has invalid characters.
function _addHostUpdateSaveBtn() {
    const type = _addHostType();
    const enabled = _addHostNameValid() && (type === 'ephemeral' || _addHostLastProbeOk);
    document.getElementById('add-host-save-btn').disabled = !enabled;
    const name = (document.getElementById('add-host-name').value || '').trim();
    const status = document.getElementById('add-host-status');
    if (name && !_addHostNameValid()) {
        status.textContent = 'Name: letters, digits, _ and - only (no spaces).';
    } else if (status.textContent.startsWith('Name:')) {
        status.textContent = '';
    }
}

// Toggle the SSH vs Ephemeral sections + button affordances. Ephemeral has
// no endpoint to probe before the VM exists, so its Test button is hidden
// and Save is enabled immediately; SSH keeps the probe-gates-Save flow.
function addHostTypeChanged() {
    const type = _addHostType();
    const isEph = type === 'ephemeral';
    document.getElementById('add-host-ssh-fields').style.display = isEph ? 'none' : 'block';
    document.getElementById('add-host-ephemeral-fields').style.display = isEph ? 'grid' : 'none';
    document.getElementById('add-host-help-ssh').style.display = isEph ? 'none' : 'block';
    document.getElementById('add-host-help-ephemeral').style.display = isEph ? 'block' : 'none';
    document.getElementById('add-host-test-btn').style.display = isEph ? 'none' : 'inline-block';
    document.getElementById('add-host-probe-results').innerHTML = '';
    document.getElementById('add-host-status').textContent = '';
    // Save requires a valid Name; SSH additionally waits for a green probe.
    _addHostUpdateSaveBtn();
    // Ephemeral hosts need the server-held Nebius connection — surface its
    // current status inline so the user knows whether spawning will work.
    if (isEph && typeof trainingFetchNebiusConnection === 'function') {
        trainingFetchNebiusConnection().then(() => trainingRefreshNebiusConnStatusLine());
    }
}

function closeAddHostDialog() {
    _abortPendingProbe();
    const overlay = document.getElementById('add-host-overlay');
    if (overlay) overlay.style.display = 'none';
}

function _addHostInvalidateProbe() {
    // Ephemeral hosts have no probe gate — editing a field must not disable
    // Save. Only the SSH flow re-arms the probe requirement.
    if (_addHostType() === 'ephemeral') return;
    _addHostLastProbeOk = false;
    document.getElementById('add-host-save-btn').disabled = true;
    document.getElementById('add-host-probe-results').innerHTML = '';
    document.getElementById('add-host-status').textContent = '';
}

function _renderProbeResults(result) {
    const el = document.getElementById('add-host-probe-results');
    const rowsHtml = result.checks.map((c) => {
        const colour = c.ok ? '#7ec699' : '#e06c75';
        const mark = c.ok ? '✓' : '✗';
        return `
          <div class="probe-check ${c.ok ? 'ok' : 'fail'}" style="display:flex; gap:8px; align-items:flex-start; padding:3px 0; font-size:11px; font-family: ui-monospace, monospace;">
            <span style="color:${colour}; min-width:16px;">${mark}</span>
            <span style="color:var(--text-primary,#ccc); min-width:64px;">${escapeHtmlSafe(c.name)}</span>
            <span style="color:var(--text-secondary,#999); flex:1;">${escapeHtmlSafe(c.detail)}</span>
          </div>`;
    }).join('');
    el.innerHTML = `
      <div style="border:1px solid var(--border,#333); border-radius:4px; padding:8px 10px; background:var(--bg-input,#2d2d30);">
        ${rowsHtml}
        <div style="margin-top:6px; font-size:10px; color:var(--text-secondary,#777);">probe completed in ${result.latency_ms} ms</div>
      </div>`;
}

function escapeHtmlSafe(s) {
    if (typeof window.escapeHtml === 'function') return window.escapeHtml(s);
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function addHostTest() {
    const host = document.getElementById('add-host-host').value.trim();
    const status = document.getElementById('add-host-status');
    const testBtn = document.getElementById('add-host-test-btn');
    const saveBtn = document.getElementById('add-host-save-btn');
    if (!host) {
        status.textContent = 'Enter the Host first (alias or user@host).';
        return;
    }
    const originalLabel = testBtn.textContent;
    testBtn.textContent = 'Testing…';
    testBtn.disabled = true;
    saveBtn.disabled = true;
    status.textContent = '';
    document.getElementById('add-host-probe-results').innerHTML = '';
    _addHostProbeAbort = new AbortController();
    try {
        const resp = await fetch('/api/training/hosts/probe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host }),
            signal: _addHostProbeAbort.signal,
        });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err.slice(0, 200)}`);
        }
        const result = await resp.json();
        _renderProbeResults(result);
        _addHostLastProbeOk = !!result.ok;
        _addHostUpdateSaveBtn();
        if (!result.ok && result.message) {
            status.textContent = result.message;
        }
    } catch (e) {
        if (e.name !== 'AbortError') {
            status.textContent = `Probe failed: ${e.message}`;
        }
        _addHostLastProbeOk = false;
        _addHostUpdateSaveBtn();
    } finally {
        testBtn.textContent = originalLabel;
        testBtn.disabled = false;
    }
}

async function addHostSubmit() {
    const name = document.getElementById('add-host-name').value.trim();
    const status = document.getElementById('add-host-status');
    const saveBtn = document.getElementById('add-host-save-btn');
    const type = _addHostType();

    if (!name) { status.textContent = 'Name is required.'; return; }
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
        status.textContent = 'Name must contain only letters, digits, _ and -.'; return;
    }

    let payload;
    if (type === 'ephemeral') {
        payload = {
            name,
            display_name: name,
            provider_id: 'nebius',
            gpu: document.getElementById('add-host-gpu').value,
            disk_gib: parseInt(document.getElementById('add-host-disk').value, 10) || 100,
            ttl_hours: parseInt(document.getElementById('add-host-ttl').value, 10) || 24,
            preemptible: document.getElementById('add-host-preemptible').checked,
        };
    } else {
        const host = document.getElementById('add-host-host').value.trim();
        if (!host) { status.textContent = 'Host is required.'; return; }
        if (!_addHostLastProbeOk) {
            status.textContent = 'Run Test first; Save is enabled after a green probe.'; return;
        }
        payload = { name, host, display_name: name };
    }

    saveBtn.disabled = true;
    status.textContent = 'Saving…';
    try {
        const resp = await fetch('/api/training/hosts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err.slice(0, 200)}`);
        }
        closeAddHostDialog();
        if (typeof window.trainingLoadHosts === 'function') {
            await window.trainingLoadHosts();
        }
        if (typeof window.showToast === 'function') {
            window.showToast('Host added', name, 'success', 4000);
        }
    } catch (e) {
        status.textContent = `Save failed: ${e.message}`;
        saveBtn.disabled = false;
    }
}

async function trainingDeleteHost(hostId, displayName) {
    const label = displayName || hostId;
    if (!confirm(`Remove host "${label}"? Saved profile will be deleted.`)) return;
    try {
        const resp = await fetch(`/api/training/hosts/${encodeURIComponent(hostId)}`, { method: 'DELETE' });
        if (!resp.ok && resp.status !== 204) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err.slice(0, 200)}`);
        }
        if (typeof window.trainingLoadHosts === 'function') {
            await window.trainingLoadHosts();
        }
        if (typeof window.showToast === 'function') {
            window.showToast('Host removed', label, 'success', 3000);
        }
    } catch (e) {
        if (typeof window.showToast === 'function') {
            window.showToast('Remove failed', e.message, 'error', 5000);
        } else {
            alert(`Remove failed: ${e.message}`);
        }
    }
}

// Invalidate the probe whenever any field changes so Save stays in sync.
// readyState guard: end-of-body scripts run before DOMContentLoaded today,
// but this also survives the script ever being loaded async/deferred.
function _attachAddHostInputListeners() {
    ['add-host-name', 'add-host-host'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => { _addHostInvalidateProbe(); _addHostUpdateSaveBtn(); });
    });
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _attachAddHostInputListeners);
} else {
    _attachAddHostInputListeners();
}

// Esc closes the dialog when it's open.
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const overlay = document.getElementById('add-host-overlay');
    if (overlay && overlay.style.display === 'flex') closeAddHostDialog();
});

window.trainingShowAddHostDialog = openAddHostDialog;
window.closeAddHostDialog = closeAddHostDialog;
window.addHostTest = addHostTest;
window.addHostTypeChanged = addHostTypeChanged;
window.addHostSubmit = addHostSubmit;
window.trainingDeleteHost = trainingDeleteHost;
