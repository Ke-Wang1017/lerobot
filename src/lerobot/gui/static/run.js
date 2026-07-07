/* LeRobot GUI - Run Tab Logic */

let runTabInitialized = false;
let runEventSource = null;
let selectedWorkflow = 'teleop'; // 'teleop' | 'replay' | 'policy'
let obsStreamMeta = null; // {available, obs_scalar_keys, action_keys, image_keys}
let obsStreamTimer = null; // interval ID for camera polling
let _runFormRendered = false; // true once all three workflow sections are in the DOM

// ============================================================================
// Initialization
// ============================================================================

async function runTabInit() {
    if (runTabInitialized) {
        // Re-check status and reconnect SSE if needed
        _toggleHvlaRecordFields();
        await pollRunStatus();
        return;
    }
    runTabInitialized = true;

    // Ensure profiles are loaded (Robot tab may not have been visited yet)
    if (typeof robotProfiles !== 'undefined' && !robotProfiles.length) {
        try {
            const res = await fetch('/api/robot/profiles');
            robotProfiles = await res.json();
        } catch (e) { /* ignore */ }
    }
    if (typeof teleopProfiles !== 'undefined' && !teleopProfiles.length) {
        try {
            const res = await fetch('/api/robot/teleop-profiles');
            teleopProfiles = await res.json();
        } catch (e) { /* ignore */ }
    }

    initSplitHandle();
    renderRunForm();
    _initSelectTitleSync();
    await _ensureModelDataLoaded();
    await pollRunStatus();

    // Backstop reconciliation: SSE events are the primary trigger for
    // re-polling, but if the stream drops at the wrong moment (server-side
    // close racing with the `done` event, browser tab backgrounded, etc.)
    // the frontend can be left thinking a long-dead subprocess is still
    // running. A low-frequency poll closes that gap without user
    // intervention; cost is one tiny GET every 5 s.
    if (!window._runStatusPollTimer) {
        window._runStatusPollTimer = setInterval(pollRunStatus, 5000);
    }
}

/** Sync select title attribute to selected option text (for ellipsis tooltip). */
function _initSelectTitleSync() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    function _syncTitle(sel) {
        const opt = sel.options[sel.selectedIndex];
        sel.title = opt ? opt.textContent : '';
    }
    sidebar.addEventListener('change', (e) => {
        if (e.target.tagName === 'SELECT') _syncTitle(e.target);
    });
    // Also observe innerHTML changes so programmatic updates get titles too
    new MutationObserver(() => {
        sidebar.querySelectorAll('select').forEach(_syncTitle);
    }).observe(sidebar, { childList: true, subtree: true });
}

// ============================================================================
// Workflow selection
// ============================================================================

async function selectWorkflow(workflow) {
    selectedWorkflow = workflow;
    document.querySelectorAll('.workflow-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.workflow === workflow);
    });
    // Toggle visibility of workflow sections (no re-render)
    for (const wf of ['teleop', 'replay', 'policy']) {
        const section = document.getElementById(`run-section-${wf}`);
        if (section) section.style.display = (wf === workflow) ? '' : 'none';
    }
    if (workflow === 'policy' || workflow === 'teleop') {
        await _ensureModelDataLoaded();
    }
    _updateLaunchButton();
}

// ============================================================================
// Launch-button validation
// ============================================================================
//
// Each workflow / policy type has its own required-field rules. We keep
// them as small per-{workflow,policy} validator functions in registries
// so adding a new policy or workflow stays local — no editing of one big
// `if/else if` chain. Each validator returns either `null` (ready) or a
// string reason (not ready, shown as the Launch button tooltip).

// Helper for validators: read trimmed value from a form input/select.
function _val(id) {
    return document.getElementById(id)?.value?.trim() || '';
}

// Per-policy-type validators. Add a new entry here when adding a new
// policy with its own required fields.
const _POLICY_VALIDATORS = {
    hvla_flow_s1: () => {
        if (!_val('run-hvla-task')) {
            return 'HVLA needs a task prompt — describe what the robot should do.';
        }
        // RLT mode (when selected) needs the Phase-1 token encoder dir;
        // without it actor.pt loads with a state-dict size mismatch.
        if (_val('run-hvla-rlt-select') && !_val('run-hvla-rlt-token-ckpt')) {
            return 'RLT mode needs an RL Token Encoder checkpoint dir.';
        }
        return null;
    },
    // Vanilla (non-HVLA) policies have no extra required fields beyond the
    // checkpoint that's already validated up-stream — no entry needed.
};

// Per-workflow validators. Each gets called only after the workflow's
// universal prerequisites (robot profile, plus workflow's own primary
// selector — episode for replay, checkpoint for policy, teleop for teleop)
// pass.
const _WORKFLOW_VALIDATORS = {
    teleop: () => {
        if (!_val('run-teleop-teleop')) return 'Select a teleop profile.';
        // Recording sub-fields apply only when a dataset is selected;
        // pure teleop runs without them.
        const datasetVal = _val('run-teleop-record-dataset');
        if (datasetVal === '') return null;
        const taskSelVal = document.getElementById('run-teleop-task-select')?.value;
        const task = (taskSelVal === '__new_task__')
            ? _val('run-teleop-task-custom')
            : (taskSelVal || '').trim();
        if (!task) return 'Recording requires a task description.';
        if (datasetVal === '__new__' && !_val('run-teleop-new-dataset-name')) {
            return 'Enter a name for the new dataset.';
        }
        return null;
    },
    replay: () => {
        if (!_val('run-replay-episode')) {
            return 'Select an episode to replay (open a dataset first if the list is empty).';
        }
        return null;
    },
    policy: () => {
        if (!_val('run-policy-checkpoint')) return 'Select a model checkpoint.';
        // Delegate to the per-policy-type registry. Picks up whatever the
        // currently selected checkpoint declares its policy_type to be.
        const policyType = _getSelectedPolicyType();
        const policyValidator = _POLICY_VALIDATORS[policyType];
        if (policyValidator) {
            const reason = policyValidator();
            if (reason) return reason;
        }
        return null;
    },
};

let _isRunning = false;

function _validateLaunch() {
    if (_isRunning) {
        return { ready: false, reason: 'A process is already running — Stop it first.' };
    }
    // Robot profile is the universal prerequisite for every workflow.
    if (!_val(_getActiveRobotSelectId())) {
        return { ready: false, reason: 'Select a robot profile.' };
    }
    const workflowValidator = _WORKFLOW_VALIDATORS[selectedWorkflow];
    if (!workflowValidator) {
        // Unknown workflow — let it through; backend will reject if invalid.
        return { ready: true, reason: '' };
    }
    const reason = workflowValidator();
    return reason ? { ready: false, reason } : { ready: true, reason: '' };
}

function _updateLaunchButton() {
    const btn = document.getElementById('run-launch-btn');
    if (!btn) return;
    const { ready, reason } = _validateLaunch();
    btn.disabled = !ready;
    // Native `title` shows on hover; gives the missing-field hint without
    // needing extra inline text below the button.
    btn.title = reason;
}

// ============================================================================
// Form rendering
// ============================================================================

function _esc(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function _robotProfileOptions() {
    // robotProfiles is global from robot.js
    if (typeof robotProfiles === 'undefined' || !robotProfiles.length) {
        return '<option value="">No robot profiles</option>';
    }
    return robotProfiles.map(p =>
        `<option value="${_esc(p.name)}">${_esc(p.name)} (${_esc(p.type || '?')})</option>`
    ).join('');
}

function _teleopProfileOptions() {
    if (typeof teleopProfiles === 'undefined' || !teleopProfiles.length) {
        return '<option value="">No teleop profiles</option>';
    }
    return teleopProfiles.map(p =>
        `<option value="${_esc(p.name)}">${_esc(p.name)} (${_esc(p.type || '?')})</option>`
    ).join('');
}

// ---- Dataset name resolution ----

/**
 * Resolve a dataset's correct repo_id by looking up its root path in sourceDatasets.
 * The source scanner's `name` field is the relative path from the source root (e.g., "thewisp/aaa"),
 * which is what the CLI expects. Falls back to the dataset's own repo_id if not found in sources.
 */
function _resolveDatasetRepoId(dataset) {
    const sd = window.sourceDatasets || {};
    for (const datasets of Object.values(sd)) {
        for (const ds of datasets) {
            if (ds.root === dataset.root) return ds.name;
        }
    }
    return dataset.repo_id;
}

// ---- Dataset / episode option helpers ----

function _episodeOptions() {
    // Build <optgroup> per opened dataset, with episodes as options
    const ds = window.datasets || {};
    const ep = window.episodes || {};
    const ids = Object.keys(ds);
    if (!ids.length) {
        return '<option value="" disabled selected>Open a dataset in the Data tab first</option>';
    }
    let html = '<option value="" disabled selected>Select episode</option>';
    for (const id of ids) {
        const d = ds[id];
        const epList = ep[id] || [];
        const label = _esc(d.repo_id || id);
        html += `<optgroup label="${label} (${epList.length} ep)">`;
        for (const e of epList) {
            const val = `${id}:${e.episode_index}`;
            html += `<option value="${_esc(val)}">${label} / Episode ${e.episode_index} (${e.length} frames)</option>`;
        }
        html += '</optgroup>';
    }
    return html;
}

function _recordDatasetOptions() {
    // None (pure teleop) + opened datasets (resume) + New dataset
    const ds = window.datasets || {};
    let html = '<option value="" selected>None (pure teleop)</option>';
    for (const id of Object.keys(ds)) {
        const d = ds[id];
        const label = d.repo_id || id;
        html += `<option value="existing:${_esc(id)}">${_esc(label)} (${d.total_episodes} ep)</option>`;
    }
    html += '<option value="__new__">+ New dataset...</option>';
    return html;
}

function _getDatasetTasks(dsId) {
    // Collect unique tasks from episodes
    const epList = (window.episodes || {})[dsId] || [];
    const tasks = [];
    const seen = new Set();
    for (const e of epList) {
        if (e.task && !seen.has(e.task)) {
            seen.add(e.task);
            tasks.push(e.task);
        }
    }
    return tasks;
}

function _findDatasetByRepoId(repoId) {
    // Find an opened dataset matching a repo_id (e.g. from train_config)
    if (!repoId) return null;
    const ds = window.datasets || {};
    for (const id of Object.keys(ds)) {
        const d = ds[id];
        if (d.repo_id === repoId || d.repo_id?.endsWith('/' + repoId)) {
            return id;
        }
    }
    return null;
}

// Track the currently selected model's training dataset info
let _policyTrainingDataset = null;  // { repo_id, root }

function _updatePolicyTaskSelector() {
    const container = document.getElementById('run-policy-task-container');
    if (!container) return;

    const info = _policyTrainingDataset;
    if (!info || !info.repo_id) {
        // No model selected or no training dataset — plain text input
        container.innerHTML = `<input type="text" id="run-policy-task" placeholder="Select a checkpoint first" value="">`;
        return;
    }

    const dsId = _findDatasetByRepoId(info.repo_id);
    if (!dsId) {
        // Training dataset not opened — show prompt to open it
        const displayName = info.repo_id.length > 40 ? '...' + info.repo_id.slice(-37) : info.repo_id;
        container.innerHTML =
            `<div class="policy-task-open-prompt">` +
            `<span class="form-hint">Training dataset not opened</span>` +
            `<button class="btn-tiny btn-accent" onclick="_openTrainingDataset()">Open ${_esc(displayName)}</button>` +
            `</div>` +
            `<input type="text" id="run-policy-task" placeholder="Or type task manually" value="">`;
        return;
    }

    // Training dataset is opened — show task dropdown + custom option
    const tasks = _getDatasetTasks(dsId);
    let html = '<select id="run-policy-task-select" onchange="_onPolicyTaskSelectChange()">';
    if (tasks.length) {
        for (const t of tasks) {
            html += `<option value="${_esc(t)}">${_esc(t)}</option>`;
        }
        html += '<option value="__new_task__">+ New task...</option>';
    } else {
        html += '<option value="__new_task__" selected>+ New task...</option>';
    }
    html += '</select>';
    html += `<input type="text" id="run-policy-task-custom" placeholder="Describe the task" style="${tasks.length ? 'display:none' : ''}" value="">`;
    container.innerHTML = html;
}

function _onPolicyTaskSelectChange() {
    const sel = document.getElementById('run-policy-task-select');
    const customInput = document.getElementById('run-policy-task-custom');
    if (!sel || !customInput) return;
    if (sel.value === '__new_task__') {
        customInput.style.display = '';
        customInput.focus();
    } else {
        customInput.style.display = 'none';
    }
}

async function _openTrainingDataset() {
    const info = _policyTrainingDataset;
    if (!info) return;
    // Open by root path if available, otherwise by repo_id
    const openPath = info.root || info.repo_id;
    if (typeof openDataset === 'function') {
        await openDataset(openPath);
        // After opening, refresh the task selector
        _updatePolicyTaskSelector();
    }
}

function _updateTaskSelect(dsId) {
    const sel = document.getElementById('run-teleop-task-select');
    const customInput = document.getElementById('run-teleop-task-custom');
    if (!sel) return;

    const tasks = dsId ? _getDatasetTasks(dsId) : [];
    let html = '';
    if (tasks.length) {
        for (const t of tasks) {
            html += `<option value="${_esc(t)}">${_esc(t)}</option>`;
        }
        html += '<option value="__new_task__">+ New task...</option>';
    } else {
        // No existing tasks — show placeholder, user types freely
        html = '<option value="__new_task__" selected>+ New task...</option>';
    }
    sel.innerHTML = html;

    // Show/hide custom input based on selection
    if (sel.value === '__new_task__') {
        if (customInput) { customInput.style.display = ''; customInput.value = ''; }
    } else {
        if (customInput) customInput.style.display = 'none';
    }
}

function _onTaskSelectChange() {
    const sel = document.getElementById('run-teleop-task-select');
    const customInput = document.getElementById('run-teleop-task-custom');
    if (!sel || !customInput) return;
    if (sel.value === '__new_task__') {
        customInput.style.display = '';
        customInput.focus();
    } else {
        customInput.style.display = 'none';
    }
}

function _onRecordDatasetChange() {
    const sel = document.getElementById('run-teleop-record-dataset');
    if (!sel) return;
    const val = sel.value;

    const newNameRow = document.getElementById('run-teleop-new-dataset-row');
    const recordFields = document.getElementById('run-teleop-record-fields');

    if (val === '') {
        // None — pure teleop
        if (newNameRow) newNameRow.style.display = 'none';
        if (recordFields) recordFields.style.display = 'none';
    } else if (val === '__new__') {
        // New dataset
        if (newNameRow) newNameRow.style.display = '';
        if (recordFields) recordFields.style.display = '';
        _updateTaskSelect(null);
        _checkNewDatasetConflict();
    } else {
        // Existing dataset — resume
        if (newNameRow) newNameRow.style.display = 'none';
        if (recordFields) recordFields.style.display = '';
        const dsId = val.replace('existing:', '');
        // Pre-fill FPS from dataset metadata
        const d = (window.datasets || {})[dsId];
        if (d) {
            const fpsInput = document.getElementById('run-teleop-fps');
            if (fpsInput) fpsInput.value = d.fps;
        }
        // Populate task selector from existing episodes
        _updateTaskSelect(dsId);
    }
}

function _checkNewDatasetConflict() {
    const nameInput = document.getElementById('run-teleop-new-dataset-name');
    const warning = document.getElementById('run-teleop-dataset-conflict');
    if (!nameInput || !warning) return;

    const name = nameInput.value.trim();
    if (!name) { warning.style.display = 'none'; return; }

    // Check across all sources
    const sd = window.sourceDatasets || {};
    for (const datasets of Object.values(sd)) {
        for (const ds of datasets) {
            if (ds.name === name) {
                warning.textContent = `"${name}" already exists in a source folder`;
                warning.style.display = '';
                return;
            }
        }
    }
    warning.style.display = 'none';
}

function _onReplayEpisodeChange() {
    const sel = document.getElementById('run-replay-episode');
    if (!sel || !sel.value) return;
    // No-op for now: replay paces at the dataset's recorded fps, which the
    // backend reads from the dataset metadata. Kept as a hook for future
    // per-episode UI (e.g. preview thumbnail when an episode is picked).
}

// ---- Policy dataset helpers ----

function _policyDatasetOptions() {
    const ds = window.datasets || {};
    let html = '<option value="__new__" selected>+ New dataset...</option>';
    for (const id of Object.keys(ds)) {
        const d = ds[id];
        const label = d.repo_id || id;
        html += `<option value="existing:${_esc(id)}">${_esc(label)} (${d.total_episodes} ep)</option>`;
    }
    return html;
}

function _onPolicyDatasetChange() {
    const sel = document.getElementById('run-policy-dataset');
    if (!sel) return;
    const val = sel.value;
    const newNameRow = document.getElementById('run-policy-new-dataset-row');
    const mainIsExisting = val.startsWith('existing:');

    if (val === '__new__') {
        if (newNameRow) newNameRow.style.display = '';
    } else {
        if (newNameRow) newNameRow.style.display = 'none';
        // Pre-fill FPS from existing dataset
        const dsId = val.replace('existing:', '');
        const d = (window.datasets || {})[dsId];
        if (d) {
            const fpsInput = document.getElementById('run-policy-fps');
            if (fpsInput) fpsInput.value = d.fps;
        }
    }

    // Update intervention dataset options: existing intervention datasets only available
    // when main dataset is also existing (both share the resume flag in the CLI)
    const intSel = document.getElementById('run-policy-intervention-dataset');
    if (intSel) {
        const prev = intSel.value;
        intSel.innerHTML = _interventionDatasetOptions(mainIsExisting);
        intSel.value = prev;
        // If previous selection is no longer valid, reset to None
        if (intSel.value !== prev) {
            intSel.value = '';
            _onInterventionDatasetChange();
        }
    }
}

async function _onPolicyTeleopChange() {
    const sel = document.getElementById('run-policy-teleop');
    const section = document.getElementById('run-policy-intervention-section');
    if (!section) return;

    if (!sel?.value) {
        section.style.display = 'none';
        return;
    }

    // Show the section but check if intervention_enabled is set in the profile
    section.style.display = '';
    const profileData = await _getProfileData('teleop', sel.value);
    const interventionEnabled = profileData?.fields?.intervention_enabled === true;
    const datasetSel = document.getElementById('run-policy-intervention-dataset');
    const nameInput = document.getElementById('run-policy-intervention-repo-id');
    if (datasetSel) {
        datasetSel.disabled = !interventionEnabled;
        if (!interventionEnabled) {
            datasetSel.value = '';
            _onInterventionDatasetChange();
        }
    }
    if (nameInput) nameInput.disabled = !interventionEnabled;
    // Show warning when intervention is not enabled
    const warning = document.getElementById('run-policy-intervention-warning');
    if (warning) warning.style.display = interventionEnabled ? 'none' : '';
}

function _interventionDatasetOptions(mainIsExisting) {
    const ds = window.datasets || {};
    let html = '<option value="" selected>None</option>';
    // Intervention resume is tied to the main dataset's resume flag in the CLI,
    // so existing intervention requires existing main, new intervention requires new main.
    if (mainIsExisting) {
        for (const id of Object.keys(ds)) {
            const d = ds[id];
            const label = d.repo_id || id;
            html += `<option value="existing:${_esc(id)}">${_esc(label)} (${d.total_episodes} ep)</option>`;
        }
    } else {
        html += '<option value="__new__">+ New dataset...</option>';
    }
    return html;
}

function _onInterventionDatasetChange() {
    const sel = document.getElementById('run-policy-intervention-dataset');
    if (!sel) return;
    const val = sel.value;
    const newNameRow = document.getElementById('run-policy-intervention-new-row');
    if (val === '__new__') {
        if (newNameRow) newNameRow.style.display = '';
    } else {
        if (newNameRow) newNameRow.style.display = 'none';
    }
}

// ---- Model / checkpoint option helpers ----

function _modelCheckpointOptions() {
    // Build options from the Model tab's cached scan data. The server
    // emits `default_policy_path` per run (api/models.py:_scan_training_run)
    // and `policy_path` per checkpoint (api/models.py:_read_checkpoint_meta);
    // we use those directly — DO NOT reconstruct the path from `m.path`.
    // Why: the layout under `m.path` differs between vanilla lerobot-train
    // (`<run>/checkpoints/<step>/pretrained_model`) and the GUI-managed
    // docker recipe (`<run>/output/checkpoints/<step>/pretrained_model`),
    // and the flat HVLA-S2-VLM layout has no `pretrained_model/` subdir
    // at all. Server-side synthesis handles all three; client-side
    // reconstruction used to silently break for the latter two.
    const data = (typeof modelSourceData !== 'undefined') ? modelSourceData : {};
    const entries = Object.values(data).flat();
    if (!entries.length) {
        return '<option value="" disabled selected>Loading models...</option>';
    }
    let html = '<option value="" disabled selected>Select checkpoint</option>';
    for (const m of entries) {
        // Defensive: an entry with no usable policy path (RLT, future
        // variants) is skipped rather than serialized as `value="undefined"`.
        if (!m.default_policy_path) continue;
        const stepText = m.current_step != null ? ` (step ${m.current_step.toLocaleString()})` : '';
        // `data-run-path` lets _onPolicyCheckpointChange recover the run
        // dir without regex-stripping the policy path (which would fail
        // for flat layouts where policy_path === run path).
        html += `<option value="${_esc(m.default_policy_path)}" data-policy-type="${_esc(m.policy_type)}" data-run-path="${_esc(m.path)}" title="${_esc(m.path)}">${_esc(m.name)} — ${_esc(m.policy_type)}${stepText}</option>`;
    }
    return html;
}

function _getDebugModelConfig() {
    // Returns debug model config or null if none selected.
    const sel = document.getElementById('run-teleop-debug-model');
    if (!sel?.value) return null;
    const opt = sel.selectedOptions[0];
    const policyType = opt?.dataset?.policyType || '';
    const config = {
        checkpoint: sel.value,
        policy_type: policyType,
    };
    if (policyType === 'hvla_s2_vlm') {
        config.task = document.getElementById('run-teleop-debug-s2-task')?.value?.trim() || '';
        config.decode_subtask = document.getElementById('run-teleop-debug-s2-decode')?.checked || false;
    }
    return config;
}

let _debugModelLoading = false;  // prevent double-click

async function _loadDebugModel() {
    if (_debugModelLoading) return;
    const config = _getDebugModelConfig();
    if (!config) {
        showToast('Error', 'Select a model first', 'error');
        return;
    }
    const status = document.getElementById('run-debug-model-status');
    _debugModelLoading = true;
    _updateDebugButtons();
    _showModelPanel(true);  // show panel immediately during loading
    if (status) status.textContent = 'Loading...';
    // Clear model terminal
    const modelTerminal = document.getElementById('run-model-terminal');
    if (modelTerminal) modelTerminal.innerHTML = '<div class="terminal-line" style="color: #666;">Loading model...</div>';
    // Connect SSE immediately so output appears as model loads
    _connectDebugOutputSSE();
    try {
        const res = await fetch('/api/run/debug/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config),
        });
        const data = await res.json();
        if (res.ok) {
            if (status) status.textContent = `Loaded (PID ${data.pid})`;
            _debugModelLoaded = true;
            showToast('Debug model', 'Model loaded', 'success');
            console.log('Debug model loaded, showing panel');
        } else {
            if (status) status.textContent = 'Not loaded';
            showToast('Error', data.detail || 'Failed to load', 'error');
        }
    } catch (e) {
        if (status) status.textContent = 'Error';
        showToast('Error', e.message, 'error');
    } finally {
        _debugModelLoading = false;
        _updateDebugButtons();
    }
}

async function _unloadDebugModel() {
    if (_debugModelLoading) return;
    const status = document.getElementById('run-debug-model-status');
    _debugModelLoading = true;
    _updateDebugButtons();
    try {
        const res = await fetch('/api/run/debug/unload', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            if (status) status.textContent = 'Not loaded';
            _debugModelLoaded = false;
            _disconnectDebugOutputSSE();
            if (data.status !== 'not_loaded') showToast('Debug model', 'Model unloaded', 'info');
        }
    } catch (e) {
        showToast('Error', e.message, 'error');
    } finally {
        _debugModelLoading = false;
        _updateDebugButtons();
    }
}

let _debugModelLoaded = false;

function _updateDebugButtons() {
    const loadBtn = document.getElementById('run-debug-load-btn');
    const unloadBtn = document.getElementById('run-debug-unload-btn');
    const sel = document.getElementById('run-teleop-debug-model');
    const hasSelection = !!sel?.value;

    // Load: enabled only when (a) not currently loading, (b) not already
    // loaded, and (c) a model is selected. Without (c) the button used to be
    // clickable and would just throw a toast — disable upfront so the
    // affordance matches the actual enabled state.
    const loadEnabled = !_debugModelLoading && !_debugModelLoaded && hasSelection;
    // Unload: enabled only when something is currently loaded (and not
    // currently loading/unloading).
    const unloadEnabled = !_debugModelLoading && _debugModelLoaded;

    if (loadBtn) {
        loadBtn.disabled = !loadEnabled;
        loadBtn.classList.toggle('btn-accent', loadEnabled);
    }
    if (unloadBtn) {
        unloadBtn.disabled = !unloadEnabled;
        unloadBtn.classList.toggle('btn-accent', unloadEnabled);
    }
    _showModelPanel(_debugModelLoaded);
}

async function _refreshDebugModelStatus() {
    const status = document.getElementById('run-debug-model-status');
    if (!status) return;
    try {
        const res = await fetch('/api/run/debug/status');
        const data = await res.json();
        _debugModelLoaded = data.loaded;
        _debugModelLoading = false;  // reset stuck loading flag on status refresh
        status.textContent = data.loaded ? `Loaded (PID ${data.pid})` : 'Not loaded';
        _updateDebugButtons();
    } catch (e) {
        status.textContent = 'Unknown';
    }
}

function _onDebugModelChange() {
    const sel = document.getElementById('run-teleop-debug-model');
    const s2Fields = document.getElementById('run-teleop-debug-s2-fields');
    if (!sel || !s2Fields) return;
    const opt = sel.selectedOptions[0];
    const policyType = opt?.dataset?.policyType || '';
    s2Fields.style.display = policyType === 'hvla_s2_vlm' ? '' : 'none';
    // Selection drives the Load button's enabled state.
    _updateDebugButtons();
}

function _getSelectedPolicyType() {
    const sel = document.getElementById('run-policy-checkpoint');
    if (!sel?.selectedOptions?.length) return '';
    return sel.selectedOptions[0].dataset.policyType || '';
}

function _updateHVLAFieldsVisibility() {
    const isHVLA = _getSelectedPolicyType() === 'hvla_flow_s1';
    const hvlaSection = document.getElementById('run-policy-hvla-fields');
    const standardSection = document.getElementById('run-policy-standard-fields');
    if (hvlaSection) hvlaSection.style.display = isHVLA ? '' : 'none';
    if (standardSection) standardSection.style.display = isHVLA ? 'none' : '';
    if (isHVLA) {
        _toggleHvlaRecordFields();
        _refreshRltCheckpoints();
    }
}

function _toggleHvlaRecordFields() {
    const hasDataset = !!document.getElementById('run-hvla-record-dataset')?.value?.trim();
    const rltVal = document.getElementById('run-hvla-rlt-select')?.value || '';
    const rltEnabled = rltVal !== '';
    const enableEpFields = hasDataset || rltEnabled;
    for (const id of ['run-hvla-episodes', 'run-hvla-episode-time', 'run-hvla-reset-time']) {
        const el = document.getElementById(id);
        if (el) el.disabled = !enableEpFields;
    }
}

function _onRltSelectChange() {
    const sel = document.getElementById('run-hvla-rlt-select');
    const val = sel?.value || '';
    const isNew = val === '__new__';
    const isExisting = val && !isNew;

    // Token encoder: shown for both New and Existing (auto-discovery may fail for old checkpoints)
    for (const id of ['run-hvla-rlt-token-label', 'run-hvla-rlt-token-ckpt']) {
        const el = document.getElementById(id);
        if (el) el.style.display = (isNew || isExisting) ? '' : 'none';
    }
    // Output dir: only for New training
    for (const id of ['run-hvla-rlt-outdir-label', 'run-hvla-rlt-output-dir']) {
        const el = document.getElementById(id);
        if (el) el.style.display = isNew ? '' : 'none';
    }
    // Experimental launch-time knobs: visible for New AND for resumed
    // training (so you can A/B the flag against your current checkpoint),
    // but NOT for deploy — there's no exploration noise to share in
    // deterministic deploy mode. Mid-training toggle does mix two noise
    // distributions in the replay buffer; the tooltip flags that.
    const isResumeTrain = isExisting &&
        (document.getElementById('run-hvla-rlt-run-mode')?.value || 'train') === 'train';
    const showShared = isNew || isResumeTrain;
    for (const id of ['run-hvla-rlt-shared-noise-label', 'run-hvla-rlt-shared-noise']) {
        const el = document.getElementById(id);
        if (el) el.style.display = showShared ? '' : 'none';
    }
    // Existing checkpoint fields
    for (const id of ['run-hvla-rlt-mode-label', 'run-hvla-rlt-run-mode']) {
        const el = document.getElementById(id);
        if (el) el.style.display = isExisting ? '' : 'none';
    }
    // Common fields (chunk length, start engaged)
    for (const id of ['run-hvla-rlt-chunk-label', 'run-hvla-rlt-chunk-length', 'run-hvla-rlt-start-label', 'run-hvla-rlt-start-engaged']) {
        const el = document.getElementById(id);
        if (el) el.style.display = (isNew || isExisting) ? '' : 'none';
    }
    _toggleHvlaRecordFields();
}

async function _refreshRltCheckpoints() {
    try {
        const res = await fetch('/api/models/rlt-checkpoints');
        if (!res.ok) return;
        const checkpoints = await res.json();
        const sel = document.getElementById('run-hvla-rlt-select');
        if (!sel) return;
        // Preserve current selection
        const currentVal = sel.value;
        // Keep None and New options, replace the rest
        sel.innerHTML = '<option value="">None (no RL)</option><option value="__new__">+ New training...</option>';
        for (const ckpt of checkpoints) {
            const label = ckpt.name + (ckpt.episode ? ` (ep ${ckpt.episode})` : '');
            const opt = document.createElement('option');
            opt.value = ckpt.path;
            opt.textContent = label;
            opt.dataset.hasReplayBuffer = ckpt.has_replay_buffer || false;
            opt.dataset.hasCritic = ckpt.has_critic || false;
            // run_dir is the parent run directory — the dir new checkpoints
            // should land in. For latest/ entries it equals path; for ep_N
            // snapshots it's the snapshot's parent so further training
            // continues into the original run dir, not into the snapshot.
            opt.dataset.runDir = ckpt.run_dir || ckpt.path;
            sel.appendChild(opt);
        }
        // Restore selection if still valid
        if (currentVal && [...sel.options].some(o => o.value === currentVal)) {
            sel.value = currentVal;
        }
    } catch (e) {
        console.warn('Failed to fetch RLT checkpoints:', e);
    }
}

function _onPolicyCheckpointChange() {
    const sel = document.getElementById('run-policy-checkpoint');
    if (!sel?.value) return;
    // Recover the run dir from the option's dataset (stashed by
    // _modelCheckpointOptions). Falls through to sel.value for flat
    // layouts where policy_path IS the run path. The old regex-strip
    // assumed `<run>/checkpoints/last/pretrained_model`, which broke
    // for both the docker recipe (extra `output/` segment) and flat
    // checkpoints (no `pretrained_model/` suffix at all).
    const runPath = sel.selectedOptions[0]?.dataset.runPath || sel.value;
    if (typeof _prefillPolicyFields === 'function') {
        _prefillPolicyFields(runPath);
    }
    _updateHVLAFieldsVisibility();
}

async function _ensureModelDataLoaded() {
    // Already have data — nothing to do
    if (typeof modelSourceData !== 'undefined' && Object.keys(modelSourceData).length) return;
    // Load sources and scan (bypass modelTabInit's one-shot guard which
    // blocks retries when the first attempt finished with empty data).
    if (typeof loadModelSources === 'function') {
        await loadModelSources();
    }
    // loadModelSources only scans sources whose `expanded` flag is true.
    // The policy tab needs models regardless of UI expansion state, so if
    // we're still empty after loadModelSources, scan every source.
    if (
        typeof modelSourceData !== 'undefined' &&
        !Object.keys(modelSourceData).length &&
        typeof modelSources !== 'undefined' &&
        modelSources.length &&
        typeof scanModelSource === 'function'
    ) {
        await Promise.all(modelSources.map(s => scanModelSource(s.path)));
    }
    // Re-render checkpoint selectors after data is loaded
    const sel = document.getElementById('run-policy-checkpoint');
    if (sel) sel.innerHTML = _modelCheckpointOptions();
    const debugSel = document.getElementById('run-teleop-debug-model');
    if (debugSel) {
        const prev = debugSel.value;
        debugSel.innerHTML = '<option value="">None</option>' + _modelCheckpointOptions();
        debugSel.value = prev;
    }
}

// ---- Refresh helpers (called from other tabs when data changes) ----

function refreshRunProfileSelects() {
    if (!_runFormRendered) return;
    const robotOpts = _robotProfileOptions();
    const teleopOpts = _teleopProfileOptions();
    for (const id of ['run-teleop-robot', 'run-replay-robot', 'run-policy-robot']) {
        const sel = document.getElementById(id);
        if (sel) { const prev = sel.value; sel.innerHTML = robotOpts; sel.value = prev; }
    }
    const teleopSel = document.getElementById('run-teleop-teleop');
    if (teleopSel) { const prev = teleopSel.value; teleopSel.innerHTML = teleopOpts; teleopSel.value = prev; }
    const policyTeleopSel = document.getElementById('run-policy-teleop');
    if (policyTeleopSel) {
        const prev = policyTeleopSel.value;
        policyTeleopSel.innerHTML = '<option value="">None (policy only)</option>' + teleopOpts;
        policyTeleopSel.value = prev;
    }
}

function refreshRunDatasetSelects() {
    if (!_runFormRendered) return;
    // Refresh replay episode selector
    const replaySel = document.getElementById('run-replay-episode');
    if (replaySel) { const prev = replaySel.value; replaySel.innerHTML = _episodeOptions(); replaySel.value = prev; }
    // Refresh teleop record dataset selector
    const recordSel = document.getElementById('run-teleop-record-dataset');
    if (recordSel) {
        const prev = recordSel.value;
        recordSel.innerHTML = _recordDatasetOptions();
        // If user was creating a new dataset, the recording is done — check if it
        // now exists as an opened dataset, otherwise reset to "None"
        if (prev === '__new__') {
            const newName = document.getElementById('run-teleop-new-dataset-name')?.value?.trim();
            let matched = false;
            if (newName) {
                const ds = window.datasets || {};
                for (const id of Object.keys(ds)) {
                    if (ds[id].repo_id === newName || ds[id].repo_id?.endsWith('/' + newName)) {
                        recordSel.value = `existing:${id}`;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) {
                recordSel.value = '';  // Reset to "None (pure teleop)"
            }
            _onRecordDatasetChange();
            return;
        }
        recordSel.value = prev;
    }
    // Refresh policy dataset selector
    const policySel = document.getElementById('run-policy-dataset');
    if (policySel) { const prev = policySel.value; policySel.innerHTML = _policyDatasetOptions(); policySel.value = prev; }
    // Refresh task selector (training dataset may have been opened/closed)
    _updatePolicyTaskSelector();
    // Refresh intervention dataset selector
    const intSel = document.getElementById('run-policy-intervention-dataset');
    if (intSel) {
        const mainVal = document.getElementById('run-policy-dataset')?.value || '__new__';
        const prev = intSel.value;
        intSel.innerHTML = _interventionDatasetOptions(mainVal.startsWith('existing:'));
        intSel.value = prev;
    }
}

// ---- Form rendering ----

// Marker for required-field labels. Visually consistent across the whole
// form — pair with the corresponding entry in `_WORKFLOW_VALIDATORS` /
// `_POLICY_VALIDATORS` so the asterisk and the Launch-button gating tell
// the same story. For conditional fields (RL Token Encoder, recording
// task / new-dataset name) the asterisk is part of the label; the
// label's `display:none` toggle naturally hides the marker too.
const _REQ = ' <span class="required-marker">*</span>';

function renderRunForm() {
    if (_runFormRendered) return; // Only build once
    _runFormRendered = true;

    const form = document.getElementById('run-form');
    if (!form) return;

    let html = '';

    // ---- Teleop workflow section ----
    html += `<div id="run-section-teleop" style="${selectedWorkflow === 'teleop' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot${_REQ}</label>`;
    html += `<select id="run-teleop-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Teleop${_REQ}</label>`;
    html += `<select id="run-teleop-teleop">${_teleopProfileOptions()}</select>`;
    html += `<label>FPS</label>`;
    html += `<input type="number" id="run-teleop-fps" value="60" min="1" max="200">`;
    html += '</div>';
    // Dataset selector (optional recording)
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Record dataset</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<select id="run-teleop-record-dataset" onchange="_onRecordDatasetChange()">${_recordDatasetOptions()}</select>`;
    html += `</div>`;
    // New dataset name (hidden by default)
    html += `<div id="run-teleop-new-dataset-row" class="form-grid" style="display:none;">`;
    html += `<label>Name${_REQ}</label>`;
    html += `<div><input type="text" id="run-teleop-new-dataset-name" placeholder="my_new_dataset" oninput="_checkNewDatasetConflict()">`;
    html += `<div class="dataset-conflict-warning" id="run-teleop-dataset-conflict" style="display:none;"></div></div>`;
    html += `</div>`;
    // Record fields (hidden when None selected)
    html += `<div id="run-teleop-record-fields" class="form-grid" style="display:none;">`;
    html += `<label>Task${_REQ}</label>`;
    html += `<div>`;
    html += `<select id="run-teleop-task-select" onchange="_onTaskSelectChange()"><option value="" selected>Pick up the cube</option></select>`;
    html += `<input type="text" id="run-teleop-task-custom" placeholder="Describe the task" style="display:none; margin-top:4px;">`;
    html += `</div>`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-teleop-num-episodes" value="50" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-teleop-episode-time" value="60" min="1">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-teleop-reset-time" value="60" min="0">`;
    html += '</div>';
    html += '</div>';
    // Debug model (optional — runs alongside teleop for live inspection of
    // model output while the human drives. Distinct from a Policy run, which
    // hands the arm to the model.)
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Debug model</div>';
    html += '<div class="form-hint" style="margin-bottom:8px;">';
    html += 'Mount a model alongside teleop so its output (text, images, ';
    html += 'subtask predictions, latents, etc.) is logged for inspection ';
    html += 'while <em>you</em> drive the arm. Unlike Policy, the model ';
    html += 'never sends actions — this is for verifying what a model ';
    html += '<em>sees</em> on real human-driven trajectories.';
    html += '</div>';
    html += '<div class="form-grid">';
    html += `<label>Model</label>`;
    html += `<select id="run-teleop-debug-model" onchange="_onDebugModelChange()">`;
    html += `<option value="">None</option>`;
    html += _modelCheckpointOptions();
    html += `</select>`;
    html += `<label></label>`;
    // Buttons start disabled; _updateDebugButtons() flips state based on
    // selection + load status. Keeps consistent with the Launch/Stop pattern:
    // exactly one is enabled+highlighted at any time.
    html += `<div><button id="run-debug-load-btn" class="btn-tiny" onclick="_loadDebugModel()" disabled>Load</button> `;
    html += `<button id="run-debug-unload-btn" class="btn-tiny" onclick="_unloadDebugModel()" disabled>Unload</button> `;
    html += `<span id="run-debug-model-status" class="form-hint">Not loaded</span></div>`;
    html += `</div>`;
    // HVLA S2 specific fields (shown when S2 checkpoint selected)
    html += `<div id="run-teleop-debug-s2-fields" style="display:none">`;
    html += '<div class="form-grid">';
    html += `<label>Task Prompt</label>`;
    html += `<input type="text" id="run-teleop-debug-s2-task" placeholder="assemble cylinder into ring" value="assemble cylinder into ring">`;
    html += `<label>Decode Subtask</label>`;
    html += `<div style="text-align:left"><input type="checkbox" id="run-teleop-debug-s2-decode" checked></div>`;
    html += '</div>';
    html += '</div>';
    html += '</div>';
    html += '</div>'; // end teleop section

    // ---- Replay workflow section ----
    html += `<div id="run-section-replay" style="${selectedWorkflow === 'replay' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot${_REQ}</label>`;
    html += `<select id="run-replay-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Episode${_REQ}</label>`;
    html += `<select id="run-replay-episode" onchange="_onReplayEpisodeChange()">${_episodeOptions()}</select>`;
    html += '</div>';
    html += '</div>'; // end replay section

    // ---- Policy workflow section ----
    html += `<div id="run-section-policy" style="${selectedWorkflow === 'policy' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot${_REQ}</label>`;
    html += `<select id="run-policy-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Checkpoint${_REQ}</label>`;
    html += `<select id="run-policy-checkpoint" onchange="_onPolicyCheckpointChange()">${_modelCheckpointOptions()}</select>`;
    // Teleop profile (optional — for manual resets between episodes)
    html += `<label>Teleop</label>`;
    html += `<div><select id="run-policy-teleop" onchange="_onPolicyTeleopChange()">`;
    html += `<option value="">None (policy only)</option>`;
    html += _teleopProfileOptions();
    html += `</select>`;
    html += `<div class="form-hint">For leader inverse following and manual resets between episodes</div></div>`;
    html += `<label>FPS</label>`;
    html += `<input type="number" id="run-policy-fps" value="30" min="1" max="200">`;
    html += '</div>';
    // ---- HVLA-specific fields (shown when HVLA checkpoint selected) ----
    html += `<div id="run-policy-hvla-fields" style="display:none">`;
    html += '<div class="form-section">';
    html += '<div class="form-section-title">HVLA Settings</div>';
    html += '<div class="form-grid">';
    html += `<label>S2 Checkpoint</label>`;
    html += `<input type="text" id="run-hvla-s2-checkpoint" placeholder="/path/to/s2/model.safetensors" value="">`;
    html += `<label>Task Prompt${_REQ}</label>`;
    html += `<input type="text" id="run-hvla-task" placeholder="assemble cylinder into ring" value="">`;
    const _hvlaDecodeSubtaskDesc = "Have S2 also emit a natural-language subtask label alongside its latent → action chunk. Useful for VLA debugging / interpretability; off by default to keep S2 inference latency minimal.";
    html += `<label title="${_hvlaDecodeSubtaskDesc}">Decode Subtask</label>`;
    html += `<input type="checkbox" id="run-hvla-decode-subtask" title="${_hvlaDecodeSubtaskDesc}">`;
    const _hvlaQueryIntervalDesc = "How often S1 (chunk policy) runs new inference, measured in policy steps. Default 2 = inference every 2 steps (15 Hz at fps=30) — small enough that motion stays responsive but big enough that S1 latency doesn't bottleneck the loop. Set to 1 for max responsiveness at higher compute cost.";
    html += `<label title="${_hvlaQueryIntervalDesc}">Query Interval (steps between S1 inference, default 2)</label>`;
    html += `<input type="number" id="run-hvla-query-interval" placeholder="2" min="0" title="${_hvlaQueryIntervalDesc}">`;
    const _hvlaDenoiseStepsDesc = "Number of flow-matching ODE solver steps per S1 inference. Higher → more accurate chunks but slower (each step is one full S1 forward pass). Default 10 is the trained-time setting; lowering to 5 cuts latency by ~50% at small accuracy cost.";
    html += `<label title="${_hvlaDenoiseStepsDesc}">Denoise Steps (default 10)</label>`;
    html += `<input type="number" id="run-hvla-denoise-steps" placeholder="10" min="1" title="${_hvlaDenoiseStepsDesc}">`;
    html += `<label title="How the policy frame is sent to the robot each tick. 'chunk' (default) packs the remaining chunk frames as an ActionChunk → predictive robots use the exact-lookup path for zero-estimation-error lookahead. 'dict' sends only the current frame → predictive robots fall back to velocity-LSQ extrapolation (rate-agnostic post-fix but still has EMA residual). Use 'dict' for A/B comparison or as rollback. Non-predictive robots are unaffected.">Action Send Shape</label>`;
    html += `<select id="run-hvla-send-action-shape" title="Default 'chunk' is the recommended best-perf path. Switch to 'dict' for A/B comparison against pre-fix behaviour.">`;
    html += `<option value="chunk">chunk (default — predictive exact-lookup)</option>`;
    html += `<option value="dict">dict (single frame — A/B / rollback)</option>`;
    html += `</select>`;
    html += `<label>Record Dataset</label>`;
    html += `<input type="text" id="run-hvla-record-dataset" placeholder="eval/hvla_eval (optional)" value="" oninput="_toggleHvlaRecordFields()">`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-hvla-episodes" value="1" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-hvla-episode-time" value="60" min="0">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-hvla-reset-time" value="20" min="0">`;
    html += `<label>Intervention Dataset</label>`;
    html += `<input type="text" id="run-hvla-intervention-dataset" placeholder="eval/hvla_interventions (optional)" value="">`;
    html += `<div class="form-hint" style="grid-column: 1 / -1;">When the human takes over via SPACE, intervention fragments are saved to this dataset.</div>`;
    html += '</div>';
    html += '</div>';
    // ---- RLT section ----
    html += '<div class="form-section">';
    html += '<div class="form-section-title">RLT Online RL</div>';
    html += '<div class="form-grid">';
    const _rltSelectDesc = "Online RL fine-tuning of the frozen S1 chunk policy via a 1×768 'RL token' bottleneck. None = pure HVLA inference. + New training = start a fresh actor/critic against the loaded S1. Existing checkpoint = resume / deploy a previously-trained actor. See project_rlt_design memory note for the broader architecture.";
    html += `<label title="${_rltSelectDesc}">RL Checkpoint</label>`;
    html += `<select id="run-hvla-rlt-select" onchange="_onRltSelectChange()" title="${_rltSelectDesc}">`;
    html += `<option value="" title="Pure HVLA inference — no RL actor in the loop.">None (no RL)</option>`;
    html += `<option value="__new__" title="Start a fresh RLT training run. Requires a Phase-1 RL-token encoder checkpoint.">+ New training...</option>`;
    html += `</select>`;
    // Fields shown when New is selected
    html += `<label id="run-hvla-rlt-token-label" style="display:none">RL Token Encoder${_REQ}</label>`;
    html += `<input type="text" id="run-hvla-rlt-token-ckpt" placeholder="outputs/rlt_token_v4_4layer_d2048/checkpoint-10000" style="display:none" title="Required: path to the trained Phase-1 RL token encoder checkpoint dir (containing encoder.pt + config.json). The actor's input dim depends on this — wrong / missing means a state_dict size mismatch crash on load.">`;
    html += `<label id="run-hvla-rlt-outdir-label" style="display:none">Output Directory</label>`;
    html += `<input type="text" id="run-hvla-rlt-output-dir" value="outputs/rlt_online" style="display:none">`;
    // Experimental knobs (NEW training only — launch-time, not runtime-tunable).
    // Architectural choices baked into the run rather than mid-training tuning.
    html += `<label id="run-hvla-rlt-shared-noise-label" style="display:none" title="Sample exploration noise once per chunk and broadcast across C frames. Smoother joint commands than per-frame iid noise. Default ON after v2_widened ep101→120 A/B (autonomous +10pp). Uncheck to A/B back to per-frame iid noise — flipping mid-training mixes noise distributions in the replay buffer for ~10 episodes.">Shared noise per chunk</label>`;
    html += `<input type="checkbox" id="run-hvla-rlt-shared-noise" checked style="display:none" title="Sample exploration noise once per chunk and broadcast across C frames. Smoother joint commands than per-frame iid noise. Default ON after v2_widened ep101→120 A/B (autonomous +10pp). Uncheck to A/B back to per-frame iid noise — flipping mid-training mixes noise distributions in the replay buffer for ~10 episodes.">`;
    // Fields shown when existing checkpoint selected
    const _rltModeDesc = "Train continues the RL gradient updates on the loaded checkpoint — actor/critic keep learning during the run. Deploy runs the actor in pure inference (no optimizer, no replay buffer writes) — use to evaluate a trained policy without polluting it with new transitions.";
    html += `<label id="run-hvla-rlt-mode-label" style="display:none" title="${_rltModeDesc}">Mode</label>`;
    html += `<select id="run-hvla-rlt-run-mode" style="display:none" onchange="_onRltSelectChange()" title="${_rltModeDesc}">`;
    html += `<option value="train" title="Continue RL training: actor + critic + replay buffer all active.">Train (continue learning)</option>`;
    html += `<option value="deploy" title="Inference only: actor weights frozen, no gradient updates, no replay buffer writes. Safe way to evaluate a trained policy.">Deploy (inference only)</option>`;
    html += `</select>`;
    // Common fields
    html += `<label id="run-hvla-rlt-chunk-label" style="display:none">Chunk Length</label>`;
    html += `<input type="number" id="run-hvla-rlt-chunk-length" value="10" min="1" max="50" style="display:none">`;
    html += `<label id="run-hvla-rlt-start-label" style="display:none">Start with RL active</label>`;
    html += `<input type="checkbox" id="run-hvla-rlt-start-engaged" checked style="display:none">`;
    html += '</div>';
    html += '</div>';
    html += '</div>';
    // ---- Standard policy fields (hidden when HVLA selected) ----
    html += `<div id="run-policy-standard-fields">`;
    // Recording settings
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Record evaluation</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<select id="run-policy-dataset" onchange="_onPolicyDatasetChange()">${_policyDatasetOptions()}</select>`;
    html += `</div>`;
    // New dataset name (shown by default since __new__ is default)
    html += `<div id="run-policy-new-dataset-row" class="form-grid">`;
    html += `<label>Name</label>`;
    html += `<input type="text" id="run-policy-repo-id" placeholder="user/eval_my_policy" value="eval/eval_policy">`;
    html += `</div>`;
    html += `<div class="form-grid">`;
    html += `<label>Task</label>`;
    html += `<div id="run-policy-task-container"><input type="text" id="run-policy-task" placeholder="Select a checkpoint first" value=""></div>`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-policy-episodes" value="10" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-policy-episode-time" value="60" min="1">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-policy-reset-time" value="60" min="0">`;
    html += '</div>';
    html += '</div>';
    // Intervention dataset (for leader inverse following — human correction fragments)
    html += '<div class="form-section" id="run-policy-intervention-section" style="display:none;">';
    html += '<div class="form-section-title">Intervention dataset</div>';
    html += '<div class="form-hint" style="margin-bottom:8px;">When the human takes over the leader arm during policy evaluation, intervention fragments are saved to a separate dataset.</div>';
    html += '<div class="form-hint dataset-conflict-warning" id="run-policy-intervention-warning" style="display:none; margin-bottom:8px;">Enable intervention_enabled in the teleop profile to use this feature.</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<select id="run-policy-intervention-dataset" onchange="_onInterventionDatasetChange()">${_interventionDatasetOptions()}</select>`;
    html += '</div>';
    html += '<div id="run-policy-intervention-new-row" class="form-grid" style="display:none;">';
    html += `<label>Name</label>`;
    html += `<input type="text" id="run-policy-intervention-repo-id" placeholder="e.g. eval/interventions" value="">`;
    html += '</div>';
    html += '</div>';
    html += '</div>'; // end standard fields
    html += '</div>'; // end policy section

    form.innerHTML = html;
    _toggleHvlaRecordFields();
    // Refresh debug-model state after the form is in the DOM. Without this,
    // both Load and Unload buttons stay in their default disabled state and
    // the user can't tell whether a model is currently loaded across re-renders.
    _refreshDebugModelStatus();
    // Re-validate the Launch button now that the form's IDs exist again,
    // and re-validate on every input/change inside the form so the button
    // tracks selections continuously rather than failing at click-time.
    form.addEventListener('input', _updateLaunchButton);
    form.addEventListener('change', _updateLaunchButton);
    _updateLaunchButton();
}

// ============================================================================
// Profile data retrieval
// ============================================================================

async function _getProfileData(kind, selectedName) {
    // If the Robot tab has this profile currently loaded, collect fresh form values
    // (the user may have unsaved edits that should be respected)
    if (typeof currentProfile !== 'undefined' && currentProfile &&
        currentProfile.kind === kind && currentProfile.name === selectedName) {
        if (typeof _collectFormFields === 'function') {
            return { ...currentProfile.data, fields: _collectFormFields() };
        }
        return currentProfile.data;
    }
    // Fetch full profile from API (list only has name + type)
    const endpoint = kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(selectedName)}`);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        return null;
    }
}

// ============================================================================
// Launch / Stop
// ============================================================================

function _getActiveRobotSelectId() {
    return `run-${selectedWorkflow}-robot`;
}

function _getActiveFpsInputId() {
    return `run-${selectedWorkflow}-fps`;
}

async function launchRun() {
    const robotSelect = document.getElementById(_getActiveRobotSelectId());
    if (!robotSelect || !robotSelect.value) {
        showToast('Error', 'Select a robot profile', 'error');
        return;
    }

    const robotData = await _getProfileData('robot', robotSelect.value);
    if (!robotData) {
        showToast('Error', `Robot profile "${robotSelect.value}" not found`, 'error');
        return;
    }

    // Cap FPS to the slowest camera's capability
    const fpsInput = document.getElementById(_getActiveFpsInputId());
    let userFps = parseInt(fpsInput?.value) || 30;
    const cams = robotData.cameras || {};
    const camFpsList = Object.values(cams).map(c => c.fps).filter(Boolean);
    if (camFpsList.length > 0) {
        const minCamFps = Math.min(...camFpsList);
        if (userFps > minCamFps) {
            fpsInput.value = minCamFps;
            userFps = minCamFps;
            showToast('FPS adjusted', `FPS capped to ${minCamFps} (camera limit)`, 'warning');
        }
    }

    let endpoint, body;

    if (selectedWorkflow === 'teleop') {
        const teleopSelect = document.getElementById('run-teleop-teleop');
        const teleopData = await _getProfileData('teleop', teleopSelect?.value);
        if (!teleopData) {
            showToast('Error', 'Select a teleop profile', 'error');
            return;
        }

        const datasetSel = document.getElementById('run-teleop-record-dataset');
        const datasetVal = datasetSel?.value || '';

        const debugModel = _getDebugModelConfig();

        if (datasetVal === '') {
            // Pure teleoperate — no recording
            endpoint = '/api/run/teleoperate';
            body = {
                robot: robotData,
                teleop: teleopData,
                fps: parseInt(document.getElementById('run-teleop-fps')?.value) || 60,
                debug_model: debugModel,
            };
        } else {
            // Record mode — existing or new dataset
            const taskSel = document.getElementById('run-teleop-task-select');
            const taskCustom = document.getElementById('run-teleop-task-custom');
            const singleTask = (taskSel?.value === '__new_task__')
                ? taskCustom?.value?.trim()
                : taskSel?.value?.trim();
            if (!singleTask) {
                showToast('Error', 'Task description is required for recording', 'error');
                return;
            }

            let repoId, root = null, resume = false;
            if (datasetVal === '__new__') {
                const name = document.getElementById('run-teleop-new-dataset-name')?.value?.trim();
                if (!name) {
                    showToast('Error', 'Enter a name for the new dataset', 'error');
                    return;
                }
                repoId = name;
            } else {
                // existing:{datasetId}
                const dsId = datasetVal.replace('existing:', '');
                const d = (window.datasets || {})[dsId];
                if (!d) {
                    showToast('Error', 'Selected dataset not found — was it closed?', 'error');
                    return;
                }
                repoId = _resolveDatasetRepoId(d);
                root = d.root;
                resume = true;

                // Warn on FPS mismatch — dataset FPS is immutable across episodes
                const currentFps = parseInt(document.getElementById('run-teleop-fps')?.value) || 30;
                if (d.fps && currentFps !== d.fps) {
                    const ok = confirm(
                        `FPS mismatch: dataset "${d.repo_id}" uses ${d.fps} FPS ` +
                        `but you selected ${currentFps} FPS.\n\n` +
                        `A dataset cannot have different FPS across episodes. ` +
                        `The recording will use ${d.fps} FPS.\n\nContinue?`
                    );
                    if (!ok) return;
                }

                // Warn on robot type mismatch
                if (d.robot_type && robotData.type && d.robot_type !== robotData.type) {
                    const ok = confirm(
                        `Robot mismatch: dataset was recorded with "${d.robot_type}" ` +
                        `but selected robot is "${robotData.type}".\n\n` +
                        `Recording with a different robot may produce incompatible data.\n\nContinue anyway?`
                    );
                    if (!ok) return;
                }
            }

            endpoint = '/api/run/record';
            body = {
                robot: robotData,
                teleop: teleopData,
                repo_id: repoId,
                root: root,
                single_task: singleTask,
                fps: parseInt(document.getElementById('run-teleop-fps')?.value) || 30,
                episode_time_s: parseFloat(document.getElementById('run-teleop-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-teleop-reset-time')?.value) || 60,
                num_episodes: parseInt(document.getElementById('run-teleop-num-episodes')?.value) || 50,
                resume: resume,
                debug_model: _getDebugModelConfig(),
            };
        }
    } else if (selectedWorkflow === 'replay') {
        const episodeSel = document.getElementById('run-replay-episode');
        if (!episodeSel?.value) {
            showToast('Error', 'Select an episode to replay', 'error');
            return;
        }
        const [dsId, epIdx] = episodeSel.value.split(':');
        const d = (window.datasets || {})[dsId];
        if (!d) {
            showToast('Error', 'Selected dataset not found — was it closed?', 'error');
            return;
        }
        // Warn if robot type doesn't match dataset
        if (d.robot_type && robotData.type && d.robot_type !== robotData.type) {
            const ok = confirm(
                `Robot mismatch: dataset was recorded with "${d.robot_type}" ` +
                `but selected robot is "${robotData.type}".\n\n` +
                `Replaying on the wrong robot can send incorrect motor commands.\n\nContinue anyway?`
            );
            if (!ok) return;
        }
        endpoint = '/api/run/replay';
        body = {
            robot: robotData,
            repo_id: _resolveDatasetRepoId(d),
            root: d.root,
            episode: parseInt(epIdx),
        };
    } else if (selectedWorkflow === 'policy') {
        const checkpointSel = document.getElementById('run-policy-checkpoint');
        if (!checkpointSel?.value) {
            showToast('Error', 'Select a model checkpoint', 'error');
            return;
        }

        const policyType = _getSelectedPolicyType();

        if (policyType === 'hvla_flow_s1') {
            // ---- HVLA dispatch ----
            const s2Ckpt = document.getElementById('run-hvla-s2-checkpoint')?.value?.trim() || null;
            const hvlaTask = document.getElementById('run-hvla-task')?.value?.trim();
            if (!hvlaTask) {
                showToast('Error', 'Task prompt is required for HVLA', 'error');
                return;
            }
            // Validate the RL Token Encoder field whenever RLT mode is selected
            // (NEW or any existing checkpoint). The actor's input dim depends
            // on the encoder's rl_token_dim; missing → state_dict size mismatch
            // crash on actor.pt load.
            {
                const rltSelEl = document.getElementById('run-hvla-rlt-select');
                const rltSelVal = rltSelEl?.value || '';
                if (rltSelVal) {
                    const tokenCkpt = document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim();
                    if (!tokenCkpt) {
                        showToast(
                            'Error',
                            'RL Token Encoder is required when RLT is enabled. ' +
                            'Set it to the trained Phase-1 encoder dir ' +
                            '(e.g. outputs/rlt_token_v4_4layer_d2048/checkpoint-10000).',
                            'error',
                        );
                        document.getElementById('run-hvla-rlt-token-ckpt')?.focus();
                        return;
                    }
                }
            }
            const recordDs = document.getElementById('run-hvla-record-dataset')?.value?.trim() || null;
            const intDs = document.getElementById('run-hvla-intervention-dataset')?.value?.trim() || null;

            // Optional teleop for intervention / inverse follow
            const teleopSelect = document.getElementById('run-policy-teleop');
            let hvlaTeleopData = null;
            if (teleopSelect?.value) {
                hvlaTeleopData = await _getProfileData('teleop', teleopSelect.value);
            }

            endpoint = '/api/run/hvla';
            body = {
                robot: robotData,
                s1_checkpoint: checkpointSel.value,
                s2_checkpoint: s2Ckpt,
                task: hvlaTask,
                fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
                s1_query_interval: parseInt(document.getElementById('run-hvla-query-interval')?.value) || null,
                denoise_steps: parseInt(document.getElementById('run-hvla-denoise-steps')?.value) || null,
                decode_subtask: document.getElementById('run-hvla-decode-subtask')?.checked || false,
                send_action_shape: document.getElementById('run-hvla-send-action-shape')?.value || 'chunk',
                record_dataset: recordDs,
                num_episodes: parseInt(document.getElementById('run-hvla-episodes')?.value) || 1,
                episode_time_s: parseFloat(document.getElementById('run-hvla-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-hvla-reset-time')?.value) || 20,
                teleop: hvlaTeleopData,
                intervention_dataset: intDs,
                ...(() => {
                    const rltSel = document.getElementById('run-hvla-rlt-select');
                    const rltVal = rltSel?.value || '';
                    if (!rltVal) return {};  // None selected
                    const rltStartEngaged = document.getElementById('run-hvla-rlt-start-engaged')?.checked !== false;
                    if (rltVal === '__new__') {
                        return {
                            rlt_mode: true,
                            rlt_token_checkpoint: document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim() || null,
                            rl_chunk_length: parseInt(document.getElementById('run-hvla-rlt-chunk-length')?.value) || 10,
                            rlt_output_dir: document.getElementById('run-hvla-rlt-output-dir')?.value?.trim() || 'outputs/rlt_online',
                            rlt_start_engaged: rltStartEngaged,
                            rlt_shared_noise_per_chunk: document.getElementById('run-hvla-rlt-shared-noise')?.checked === true,
                        };
                    }
                    // Existing checkpoint
                    const runMode = document.getElementById('run-hvla-rlt-run-mode')?.value || 'train';
                    // For ep_N snapshot entries, run_dir points to the parent
                    // run dir so continued training writes back into it
                    // rather than nesting inside the snapshot. For latest/
                    // entries run_dir equals path.
                    const selectedOpt = rltSel.options[rltSel.selectedIndex];
                    const runDir = selectedOpt?.dataset?.runDir || rltVal;
                    return {
                        rlt_mode: true,
                        rlt_checkpoint: rltVal,  // path to existing actor.pt directory
                        rlt_token_checkpoint: document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim() || null,
                        rlt_deploy: runMode === 'deploy',
                        rl_chunk_length: parseInt(document.getElementById('run-hvla-rlt-chunk-length')?.value) || 10,
                        rlt_output_dir: runDir,
                        rlt_start_engaged: rltStartEngaged,
                        // Only meaningful in train mode (deploy uses deterministic actor)
                        rlt_shared_noise_per_chunk: runMode === 'train'
                            && document.getElementById('run-hvla-rlt-shared-noise')?.checked === true,
                    };
                })(),
            };
        } else {
            // ---- Standard policy dispatch ----
            // Get task from dropdown or custom input
            const taskSel = document.getElementById('run-policy-task-select');
            const taskCustom = document.getElementById('run-policy-task-custom');
            const taskPlain = document.getElementById('run-policy-task');
            let task;
            if (taskSel) {
                task = (taskSel.value === '__new_task__')
                    ? taskCustom?.value?.trim()
                    : taskSel.value?.trim();
            } else {
                task = taskPlain?.value?.trim();
            }
            if (!task) {
                showToast('Error', 'Task description is required', 'error');
                return;
            }

            // Resolve dataset: new or existing
            const datasetSel = document.getElementById('run-policy-dataset');
            const datasetVal = datasetSel?.value || '__new__';
            let repoId, root = null, resume = false;

            if (datasetVal === '__new__') {
                repoId = document.getElementById('run-policy-repo-id')?.value?.trim();
                if (!repoId) {
                    showToast('Error', 'Enter a dataset name for evaluation recordings', 'error');
                    return;
                }
            } else {
                const dsId = datasetVal.replace('existing:', '');
                const d = (window.datasets || {})[dsId];
                if (!d) {
                    showToast('Error', 'Selected dataset not found — was it closed?', 'error');
                    return;
                }
                repoId = _resolveDatasetRepoId(d);
                root = d.root;
                resume = true;
            }

            // Optional teleop for manual resets
            const teleopSelect = document.getElementById('run-policy-teleop');
            let teleopData = null;
            if (teleopSelect?.value) {
                teleopData = await _getProfileData('teleop', teleopSelect.value);
            }

            // Intervention dataset (for leader inverse following correction data)
            let interventionRepoId = null;
            const interventionSel = document.getElementById('run-policy-intervention-dataset');
            const interventionVal = interventionSel?.value || '';
            if (interventionVal === '__new__') {
                interventionRepoId = document.getElementById('run-policy-intervention-repo-id')?.value?.trim() || null;
                if (!interventionRepoId) {
                    showToast('Error', 'Enter a name for the intervention dataset', 'error');
                    return;
                }
            } else if (interventionVal.startsWith('existing:')) {
                const intDsId = interventionVal.replace('existing:', '');
                const intDs = (window.datasets || {})[intDsId];
                if (intDs) interventionRepoId = _resolveDatasetRepoId(intDs);
            }

            endpoint = '/api/run/record';
            body = {
                robot: robotData,
                teleop: teleopData,
                policy_path: checkpointSel.value,
                repo_id: repoId,
                root: root,
                single_task: task,
                fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
                episode_time_s: parseFloat(document.getElementById('run-policy-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-policy-reset-time')?.value) || 60,
                num_episodes: parseInt(document.getElementById('run-policy-episodes')?.value) || 10,
                resume: resume,
                intervention_repo_id: interventionRepoId,
            };
        }
    } else {
        showToast('Error', 'Unknown workflow', 'error');
        return;
    }

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            showToast('Launch failed', err.detail || 'Unknown error', 'error');
            return;
        }
        const data = await res.json();
        showToast('Started', `${data.command} started (PID ${data.pid})`, 'success');
        updateRunUI(true);
        connectOutputSSE();
        if (body?.rlt_mode) {
            _startRLTPoll();
            // Remember the output dir of a freshly-launched "+ New training..." run
            // so we can auto-select its new checkpoint in the dropdown when the
            // run ends (see connectOutputSSE done handler).
            _pendingNewRltOutputDir = body.rlt_checkpoint ? null : (body.rlt_output_dir || null);
        }
        // Latency dashboard polls regardless of workflow — the teleop subprocess
        // always emits latency_snapshot.json, and the endpoint returns an empty
        // stub when the file isn't there yet, so this is safe across workflows.
        _startLatencyPoll();
        // Start live camera viewer (polls until obs stream is available)
        startObsStreamViewer();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function stopRun() {
    try {
        const res = await fetch('/api/run/stop', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            showToast('Error', err.detail || 'Failed to stop', 'error');
            return;
        }
        showToast('Stopped', 'Process stopped', 'info');
        stopObsStreamViewer();
        updateRunUI(false);
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

// ============================================================================
// SSE output streaming
// ============================================================================

function connectOutputSSE() {
    disconnectOutputSSE();

    const terminal = document.getElementById('run-terminal');
    if (terminal) terminal.innerHTML = '';

    // Reset table deduplication state
    _lastTableBlock = null;
    _tableBlockNode = null;

    runEventSource = new EventSource('/api/run/output');

    runEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.done) {
            disconnectOutputSSE();
            _stopRLTPoll();
            _stopLatencyPoll();
            // Sliders only make sense during an active run — disable
            // them. The diag button is sticky (preference persists in
            // localStorage), so just refresh its tooltip to reflect the
            // staged-for-next-launch semantics.
            _setRLTSlidersEnabled(false);
            const diagOn = document.getElementById('rlt-btn-diagnostic')
                ?.classList.contains('active') || false;
            _applyDiagBtnVisual(diagOn, false);
            pollRunStatus();
            // Re-scan dataset sources and refresh opened datasets
            if (typeof window.refreshExpandedSources === 'function') {
                window.refreshExpandedSources();
            }
            if (typeof window.refreshOpenedDatasets === 'function') {
                window.refreshOpenedDatasets();
            }
            // Refresh the RLT checkpoint dropdown unconditionally after any
            // RLT run so the displayed episode count reflects the latest
            // saved checkpoint, not the value from when the dropdown was
            // first populated. After "+ New training..." also auto-select
            // the newly-written run dir so the next Start naturally resumes
            // from it instead of silently re-starting a fresh training.
            const newOutDir = _pendingNewRltOutputDir;
            _pendingNewRltOutputDir = null;
            _refreshRltCheckpoints().then(() => {
                if (!newOutDir) return;
                const sel = document.getElementById('run-hvla-rlt-select');
                if (!sel) return;
                // Paths in the dropdown are absolute; the user's output dir
                // may be relative — match by suffix in either direction.
                const match = [...sel.options].find(o => {
                    if (!o.value || o.value === '__new__') return false;
                    return o.value === newOutDir
                        || o.value.endsWith('/' + newOutDir)
                        || newOutDir.endsWith('/' + o.value);
                });
                if (match) {
                    sel.value = match.value;
                    sel.dispatchEvent(new Event('change'));
                }
            });
            return;
        }
        if (data.overlay) {
            const el = document.getElementById('debug-model-overlay');
            if (el) {
                el.textContent = data.overlay.text || '';
                el.style.color = data.overlay.color || '#fff';
                el.style.display = data.overlay.text ? 'block' : 'none';
            }
        }
        if (data.line !== undefined) {
            appendTerminalLine(data.line);
        }
    };

    runEventSource.onerror = () => {
        // SSE can drop for many reasons: server-side close after a `done`
        // event raced with the connection teardown, network blip, browser
        // backgrounding the tab, etc. If we don't reconcile here the
        // frontend can stay locked at _isRunning=true forever even though
        // the subprocess already exited — Stop AND Launch both end up
        // greyed out and the user has to refresh the page to unstick.
        disconnectOutputSSE();
        pollRunStatus();
    };
}

function disconnectOutputSSE() {
    if (runEventSource) {
        runEventSource.close();
        runEventSource = null;
    }
}

// Strip ANSI escape codes (cursor movement, colors, etc.)
function _stripAnsi(text) {
    return text.replace(/\x1b\[[0-9;]*[A-Za-z]|\[\d+[A-Za-z]/g, '');
}

// Classify a terminal line for styling
function _classifyLine(text, isStderr) {
    // System messages from our wrapper
    if (text.startsWith('---')) return 'system';
    if (text.startsWith('$')) return 'system';
    if (!isStderr) return '';
    // Classify stderr by log level
    if (/\bERROR\b/i.test(text)) return 'stderr-error';
    if (/\bWARN(?:ING)?\b/i.test(text)) return 'stderr-warn';
    // Most stderr is just INFO logging — use subdued style
    return 'stderr-info';
}

// Detect repeating table blocks (teleop uses cursor-up ANSI to overwrite)
let _lastTableBlock = null;
let _tableBlockNode = null;

let _debugOutputSSE = null;

function _showModelPanel(show) {
    const panel = document.getElementById('run-terminal-panel-model');
    if (!panel) { console.warn('Model panel element not found'); return; }
    panel.style.display = show ? '' : 'none';
    console.log('Model panel display:', show ? 'visible' : 'hidden');
    if (show && !_debugOutputSSE) {
        _connectDebugOutputSSE();
    } else if (!show && _debugOutputSSE) {
        _disconnectDebugOutputSSE();
    }
}

function _connectDebugOutputSSE() {
    _disconnectDebugOutputSSE();
    _debugOutputSSE = new EventSource('/api/run/debug/output');
    _debugOutputSSE.onmessage = (event) => {
        const modelTerminal = document.getElementById('run-model-terminal');
        if (!modelTerminal) return;
        const text = _stripAnsi(event.data);
        const line = document.createElement('div');
        line.className = 'terminal-line';
        line.textContent = text;
        modelTerminal.appendChild(line);
        modelTerminal.scrollTop = modelTerminal.scrollHeight;
        while (modelTerminal.children.length > 500) {
            modelTerminal.removeChild(modelTerminal.firstChild);
        }

    };
}

// TODO: generalize — currently hardcoded for HVLA S2 subtask text.
// When multiple debug model types exist, the model should declare its
// output schema and the user picks which field to overlay.

// Called from the camera poll loop (startObsStreamViewer) — no separate timer.
async function _pollSubtaskOverlay() {
    if (!_debugModelLoaded) return;
    try {
        const res = await fetch('/api/run/debug/subtask');
        if (!res.ok) return;
        const data = await res.json();
        const overlay = document.getElementById('debug-model-overlay');
        if (!overlay) return;
        if (data.subtask) {
            const conf = data.confidence != null ? ` (${(data.confidence * 100).toFixed(0)}%)` : '';
            overlay.textContent = data.subtask + conf;
            overlay.style.display = 'block';
        }
    } catch (e) { /* ignore */ }
}

function _hideSubtaskOverlay() {
    const overlay = document.getElementById('debug-model-overlay');
    if (overlay) {
        overlay.style.display = 'none';
        overlay.textContent = '';
    }
}

function _disconnectDebugOutputSSE() {
    if (_debugOutputSSE) {
        _debugOutputSSE.close();
        _debugOutputSSE = null;
    }
    _hideSubtaskOverlay();
}

function copyTerminal(terminalId) {
    const terminal = document.getElementById(terminalId);
    if (!terminal) return;
    const lines = Array.from(terminal.querySelectorAll('.terminal-line'))
        .map(el => el.textContent).join('\n');
    _copyTextToClipboard(lines).then(ok => {
        if (ok) showToast('Terminal', 'Copied to clipboard', 'info');
        else showToast('Terminal', 'Copy failed', 'error');
    });
}

// Clipboard helper with fallback for non-secure contexts (HTTP over LAN IP).
// navigator.clipboard is only defined on HTTPS or localhost; otherwise fall
// back to a hidden textarea + document.execCommand('copy').
function _copyTextToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).then(() => true).catch(() => false);
    }
    return new Promise(resolve => {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '0';
        ta.style.left = '0';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        resolve(ok);
    });
}

function clearTerminal(terminalId) {
    const terminal = document.getElementById(terminalId);
    if (!terminal) return;
    terminal.innerHTML = '';
}

function appendTerminalLine(rawText) {
    const terminal = document.getElementById('run-terminal');
    if (!terminal) return;

    const isStderr = rawText.startsWith('[stderr] ');
    const text = _stripAnsi(isStderr ? rawText.slice(9) : rawText);

    // Skip empty lines from ANSI stripping
    if (!text.trim()) return;

    // Detect repeating table output (teleop prints the same table repeatedly)
    // The table starts with a long "---" separator and ends with a "loop time" line.
    // We collapse repeated table blocks into a single DOM node updated in-place.
    if (!isStderr && text.match(/^-{10,}$/)) {
        _lastTableBlock = [text];
        return;
    }
    if (_lastTableBlock !== null) {
        _lastTableBlock.push(text);
        if (/loop.time/i.test(text)) {
            const blockText = _lastTableBlock.join('\n');
            if (_tableBlockNode && _tableBlockNode.parentNode === terminal) {
                _tableBlockNode.textContent = blockText;
            } else {
                _tableBlockNode = document.createElement('div');
                _tableBlockNode.className = 'terminal-line';
                _tableBlockNode.textContent = blockText;
                terminal.appendChild(_tableBlockNode);
            }
            _lastTableBlock = null;
            terminal.scrollTop = terminal.scrollHeight;
            return;
        }
        // Safety: flush if accumulator grows too large (missed terminator)
        if (_lastTableBlock.length > 50) {
            console.warn('Table block accumulator overflow, flushing', _lastTableBlock.length, 'lines');
            _lastTableBlock = null;
        } else {
            return;
        }
    }

    const line = document.createElement('div');
    line.className = 'terminal-line';
    const cls = _classifyLine(text, isStderr);
    if (cls) line.classList.add(cls);
    line.textContent = text;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;

    // Cap DOM nodes
    while (terminal.children.length > 2000) {
        terminal.removeChild(terminal.firstChild);
    }
}

// ============================================================================
// Status polling
// ============================================================================

async function pollRunStatus() {
    try {
        const res = await fetch('/api/run/status');
        const status = await res.json();
        updateRunUI(status.running);

        // If running but no SSE, reconnect
        if (status.running && !runEventSource) {
            connectOutputSSE();
            startObsStreamViewer();
        }
    } catch (e) {
        console.error('Status poll failed:', e);
    }
}

// ============================================================================
// UI state management
// ============================================================================

function updateRunUI(isRunning) {
    _isRunning = isRunning;  // mirror for _validateLaunch
    const stopBtn = document.getElementById('run-stop-btn');
    const formInputs = document.querySelectorAll('#run-form input, #run-form select');
    const workflowBtns = document.querySelectorAll('.workflow-btn');

    if (stopBtn) stopBtn.disabled = !isRunning;

    formInputs.forEach(el => el.disabled = isRunning);
    workflowBtns.forEach(el => el.disabled = isRunning);

    const indicator = document.getElementById('run-status-indicator');
    if (indicator) {
        indicator.textContent = isRunning ? 'Running' : 'Idle';
        indicator.className = isRunning ? 'run-status running' : 'run-status idle';
    }
    // Launch button is now state-driven via _updateLaunchButton (running
    // state + form validity). Don't unconditionally disable it here —
    // _validateLaunch handles the "running" case via _isRunning.
    _updateLaunchButton();
}

// ============================================================================
// Resizable split between camera viewer and terminal
// ============================================================================

function initSplitHandle() {
    const handle = document.getElementById('run-split-handle');
    const viewer = document.getElementById('rerun-viewer');
    const termContainer = document.getElementById('run-terminal-container');
    if (!handle || !viewer || !termContainer) return;

    let startY, startViewerH, startTermH;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        startY = e.clientY;
        startViewerH = viewer.offsetHeight;
        startTermH = termContainer.offsetHeight;
        handle.classList.add('dragging');

        function onMove(e) {
            const dy = e.clientY - startY;
            const newViewerH = Math.max(100, startViewerH + dy);
            const newTermH = Math.max(60, startTermH - dy);
            viewer.style.flex = 'none';
            viewer.style.height = newViewerH + 'px';
            termContainer.style.flex = 'none';
            termContainer.style.height = newTermH + 'px';
        }

        function onUp() {
            handle.classList.remove('dragging');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}

// ============================================================================
// Live camera viewer (obs-stream via shared memory)
// ============================================================================

async function startObsStreamViewer() {
    stopObsStreamViewer();

    const container = document.getElementById('rerun-viewer');
    if (!container) return;

    // Poll until the stream becomes available (robot needs time to connect)
    let attempts = 0;
    const maxAttempts = 30; // 30 × 500ms = 15s
    while (attempts < maxAttempts) {
        try {
            const res = await fetch('/api/run/obs-stream/meta');
            obsStreamMeta = await res.json();
            if (obsStreamMeta?.available) break;
        } catch (e) { /* not ready yet */ }
        await new Promise(r => setTimeout(r, 500));
        attempts++;
    }

    // Probe the URDF visualization — it's on by default, appearing as one
    // grid tile whenever the robot has a vendored URDF (see /api/run/urdf-viz).
    let urdfVizActive = false;
    try {
        const res = await fetch('/api/run/urdf-viz');
        urdfVizActive = !!(await res.json()).available;
    } catch (e) { /* probe failed; leave inactive */ }

    if (!obsStreamMeta?.available && !urdfVizActive) {
        console.warn('Neither obs-stream nor urdf-viz available after timeout');
        return;
    }

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    // Remove any old content (iframe or previous grid)
    const oldIframe = container.querySelector('iframe');
    if (oldIframe) oldIframe.remove();
    let grid = container.querySelector('.obs-cam-grid');
    if (grid) grid.remove();

    // Build the tile grid: cameras + (optionally) the URDF viz tile.
    const camKeys = obsStreamMeta?.available ? Object.keys(obsStreamMeta.image_keys) : [];
    if (camKeys.length === 0 && !urdfVizActive) return;

    const tileCount = camKeys.length + (urdfVizActive ? 1 : 0);
    grid = document.createElement('div');
    grid.className = 'obs-cam-grid';
    const cols = tileCount <= 2 ? tileCount : Math.min(tileCount, 3);
    const rows = Math.max(1, Math.ceil(tileCount / cols));
    grid.style.cssText = `
        display: grid;
        grid-template-columns: repeat(${cols}, 1fr);
        grid-template-rows: repeat(${rows}, 1fr);
        gap: 4px;
        width: 100%; height: 100%;
        padding: 4px;
        box-sizing: border-box;
    `;

    const imgElements = {};
    for (const key of camKeys) {
        const cell = document.createElement('div');
        cell.style.cssText = 'position: relative; overflow: hidden; background: #111; border-radius: 4px;';

        const img = document.createElement('img');
        img.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
        img.alt = key;
        cell.appendChild(img);

        const label = document.createElement('div');
        label.textContent = key;
        label.style.cssText = `
            position: absolute; top: 4px; left: 6px;
            color: #ccc; font-size: 11px; font-family: monospace;
            background: rgba(0,0,0,0.5); padding: 1px 5px; border-radius: 3px;
        `;
        cell.appendChild(label);

        // Per-camera latency overlay. Populated by _updateCameraOverlays
        // when the latency snapshot includes this camera (cam_* stages
        // merged across all fresh tracks). The short name (last
        // dot-separated segment) is what the tracer uses (cam_<short>_*).
        const shortName = _camShortName(key);
        const overlay = document.createElement('div');
        overlay.className = 'cam-latency-overlay';
        overlay.dataset.camShort = shortName;
        overlay.style.display = 'none';
        cell.appendChild(overlay);

        grid.appendChild(cell);
        imgElements[key] = img;
    }

    // URDF visualization tile — the in-browser three.js/urdf-loader viewer,
    // served same-origin (no separate process or port).
    if (urdfVizActive) {
        const cell = document.createElement('div');
        cell.style.cssText = 'position: relative; overflow: hidden; background: #111; border-radius: 4px;';
        const iframe = document.createElement('iframe');
        // `?urdfGhost=on` on the parent URL propagates as the iframe's
        // initial ghost state — bookmarkable, and what the screenshot
        // script keys off.
        const ghostInit = new URLSearchParams(location.search).get('urdfGhost') === 'on' ? '&ghost=on' : '';
        iframe.src = `/static/urdf_viz.html?v=4${ghostInit}`;
        iframe.style.cssText = 'width: 100%; height: 100%; border: none; background: #1a1a1a;';
        iframe.title = 'Robot visualizer';
        cell.appendChild(iframe);
        const label = document.createElement('div');
        label.textContent = 'visualizer';
        label.style.cssText = `
            position: absolute; top: 4px; left: 6px;
            color: #ccc; font-size: 11px; font-family: monospace;
            background: rgba(0,0,0,0.5); padding: 1px 5px; border-radius: 3px;
            pointer-events: none;
        `;
        cell.appendChild(label);
        grid.appendChild(cell);
    }

    container.appendChild(grid);

    // Poll camera frames at ~10fps
    let frameSeq = 0;
    obsStreamTimer = setInterval(() => {
        const seq = ++frameSeq;
        for (const key of camKeys) {
            const img = imgElements[key];
            if (!img) continue;
            // Append seq to bust browser cache
            img.src = `/api/run/obs-stream/image/${encodeURIComponent(key)}?_=${seq}`;
        }
        _pollSubtaskOverlay();
    }, 50);
}

function stopObsStreamViewer() {
    if (obsStreamTimer) {
        clearInterval(obsStreamTimer);
        obsStreamTimer = null;
    }
    obsStreamMeta = null;
    _hideSubtaskOverlay();

    const container = document.getElementById('rerun-viewer');
    if (!container) return;
    const grid = container.querySelector('.obs-cam-grid');
    if (grid) grid.remove();

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = '';
}


// =============================================================================
// RLT Training Dashboard
// =============================================================================

let _rltPollTimer = null;
// Output dir of the most recent "+ New training..." RLT launch. When the
// run ends we refresh the checkpoint dropdown and auto-select the entry
// that matches this path, so the next launch resumes from it instead of
// starting a fresh training.
let _pendingNewRltOutputDir = null;

function _switchBottomTab(tab) {
    document.querySelectorAll('.run-bottom-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === tab);
    });
    const outputPanel = document.getElementById('run-bottom-panel-output');
    const rlPanel = document.getElementById('run-bottom-panel-rl');
    const latencyPanel = document.getElementById('run-bottom-panel-latency');
    if (outputPanel) outputPanel.style.display = (tab === 'output') ? '' : 'none';
    if (rlPanel) rlPanel.style.display = (tab === 'rl') ? '' : 'none';
    if (latencyPanel) latencyPanel.style.display = (tab === 'latency') ? '' : 'none';
}

function _showRLTab(show) {
    const tab = document.getElementById('run-bottom-tab-rl');
    if (tab) tab.style.display = show ? '' : 'none';
}

function _startRLTPoll() {
    _stopRLTPoll();
    _showRLTab(true);
    _rltPollTimer = setInterval(_fetchRLTMetrics, 2000);
    _fetchRLTMetrics();
    _initRLTSliders();
}

function _stopRLTPoll() {
    if (_rltPollTimer) { clearInterval(_rltPollTimer); _rltPollTimer = null; }
}

// =============================================================================
// Latency dashboard — polls /api/run/latency-sources at 1 Hz to discover
// which loops are publishing snapshots, then fetches each fresh source and
// renders it as its own track. Multi-thread loops (e.g. HVLA's main +
// inference threads) thus appear as two stacked tracks of the same process;
// single-thread loops (teleop, record) render as one. The track DOM block
// is built from a template per source key — the rendering functions here
// take a ``scope`` element and query data-role children inside it, so the
// same code drives one track or many.
// =============================================================================

let _latencyPollTimer = null;
// Per-track render state keyed by source key (e.g. "teleop", "hvla_main").
// Each entry holds the latest iterations dict (for the percentile picker
// to re-render without a fresh fetch) and the only-widen Gantt X range
// (so per-iteration jitter doesn't move the axis bounds visibly).
const _trackStates = new Map();

function _showLatencyTab(show) {
    const tab = document.getElementById('run-bottom-tab-latency');
    if (tab) tab.style.display = show ? '' : 'none';
}

// Camera key in the obs stream is the full feature path (e.g. "top" or
// "observation.images.top"); the latency tracer uses just the last
// dot-segment to build "cam_<short>_stale_ms" / "cam_<short>_period_ms".
// This must match the cam_key used by the robot's get_observation().
function _camShortName(featureKey) {
    const idx = featureKey.lastIndexOf('.');
    return idx >= 0 ? featureKey.slice(idx + 1) : featureKey;
}

// Color palette for span bars in the Gantt. Picked to be distinct without
// being garish; consistent across iterations so the operator builds muscle
// memory ("the wide green bar is process_action").
const _GANTT_SPAN_COLORS = {
    // Teleop / record
    get_observation: '#4fc3f7',
    process_obs:     '#9b59b6',
    process_action:  '#2ecc71',
    action_send:     '#e67e22',
    infer_total:     '#e74c3c',
    inference:       '#e74c3c',
    infer_forward:   '#e74c3c',  // legacy V2 doc name; kept for back-compat
    dataset_write:   '#95a5a6',
    video_encode:    '#7f8c8d',
    // HVLA main loop
    publish_obs:     '#9b59b6',
    get_chunk:       '#f1c40f',
    // HVLA inference thread (GPU stages, sequential on the CUDA stream)
    batch_prep:      '#3498db',
    enc_obs:         '#5dade2',
    rl_tok:          '#af7ac5',
    s1_denoise:      '#e74c3c',
    actor:           '#f39c12',
};
const _GANTT_DEFAULT_SPAN_COLOR = '#aaaaaa';

// Snap the X-axis range to multiples of this many ms so small per-iteration
// jitter doesn't keep redrawing the timeline at slightly different scales.
const _GANTT_AXIS_STEP_MS = 10;

// Pick a "nice" tick interval (1, 2, 5, 10, 20, 50, ... ms) that produces
// roughly `targetCount` ticks across `range_ms`. Standard chart-axis trick:
// scale to a power of 10, snap to {1, 2, 5}, scale back. Avoids ugly tick
// values like "13.7 ms" between gridlines.
function _niceTickStep(range_ms, targetCount) {
    const raw = range_ms / Math.max(targetCount, 1);
    const exp = Math.floor(Math.log10(raw));
    const base = Math.pow(10, exp);
    const norm = raw / base;
    let nice;
    if (norm <= 1) nice = 1;
    else if (norm <= 2) nice = 2;
    else if (norm <= 5) nice = 5;
    else nice = 10;
    return nice * base;
}

// One-time outliers (torch.compile autotune, GC pause, etc.) should NOT
// stretch the shared Gantt axis. A single 20-second compile spike at
// step=0 would otherwise squish every normal iteration's bars (10-90 ms)
// into a sub-pixel sliver against the left edge. Cap any iteration's
// contribution to the union at this many times the p99 reference. The
// outlier card itself still renders fine on its own when explicitly
// selected (the cap is only on the SHARED axis range).
const _GANTT_OUTLIER_MULTIPLIER = 10;
// Absolute floor for the cap — when p99 is very small (warming up, few
// samples), don't let the multiplier shrink the axis below something
// reasonable for a 30 Hz loop.
const _GANTT_OUTLIER_FLOOR_MS = 200;

function _computeStableRange(iterations) {
    // Pessimistic union over the representative iterations so switching
    // between median/p95/max in the dropdown doesn't rescale the axis —
    // EXCEPT we cap each iteration's contribution against a reference
    // derived from p99/p95 so a compile spike can't pollute the axis.
    const ref = iterations.p99?.loop_dt_ms ?? iterations.p95?.loop_dt_ms ?? 0;
    const cap = Math.max(_GANTT_OUTLIER_FLOOR_MS, ref * _GANTT_OUTLIER_MULTIPLIER);

    let minMs = 0;
    let maxMs = 0;
    for (const rec of Object.values(iterations)) {
        if (!rec) continue;
        // Skip records whose loop_dt is wildly above the typical iteration
        // — they're one-time outliers (compile, eviction, GC) and surfacing
        // them via the shared axis would squish every other bar. Their own
        // card / dropdown selection still works because _drawGantt uses
        // whatever range we pass and a single-record render isn't subject
        // to this filter.
        if ((rec.loop_dt_ms || 0) > cap) continue;
        if ((rec.loop_dt_ms || 0) > maxMs) maxMs = rec.loop_dt_ms;
        for (const [, [s, e]] of Object.entries(rec.spans || {})) {
            if (s < minMs) minMs = s;
            if (e > maxMs) maxMs = e;
        }
        for (const ev of Object.values(rec.cam_events || {})) {
            if (ev.captured_at_ms < minMs) minMs = ev.captured_at_ms;
            if (ev.consumed_at_ms > maxMs) maxMs = ev.consumed_at_ms;
        }
    }
    // Round to a coarse grid so small jitter doesn't move the bounds.
    const step = _GANTT_AXIS_STEP_MS;
    minMs = Math.floor(minMs / step) * step;
    maxMs = Math.ceil(maxMs / step) * step;
    if (maxMs - minMs < step) maxMs = minMs + step;
    return [minMs, maxMs];
}

function _renderGantt(scope, iterations, state) {
    state.latestIterations = iterations;
    const select = scope.querySelector('[data-role="gantt-pick"]');
    const which = (select && select.value) || 'aggregate_median';
    const rec = iterations[which];
    const empty = scope.querySelector('[data-role="gantt-empty"]');
    const canvas = scope.querySelector('[data-role="gantt-canvas"]');
    if (!canvas) return;

    if (!rec || !rec.spans || Object.keys(rec.spans).length === 0) {
        if (empty) empty.style.display = '';
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        const meta = scope.querySelector('[data-role="gantt-meta"]');
        if (meta) meta.textContent = '';
        const legend = scope.querySelector('[data-role="gantt-legend"]');
        if (legend) legend.innerHTML = '';
        return;
    }
    if (empty) empty.style.display = 'none';

    const meta = scope.querySelector('[data-role="gantt-meta"]');
    if (meta) {
        if (rec.synthetic) {
            const p = rec.percentile != null ? `p${Math.round(rec.percentile)}` : 'aggregate';
            const n = rec.n_aggregated || 0;
            meta.textContent = `${p} of ${n} iters · loop ${(rec.loop_dt_ms || 0).toFixed(1)} ms`;
        } else {
            meta.textContent = `step ${rec.step ?? '?'} · loop ${(rec.loop_dt_ms || 0).toFixed(1)} ms`;
        }
    }

    // Stable range: compute from all reps, then take the union with the
    // previous render's range (only widen, never shrink). Keeps the axis
    // visually stable even as new outliers stretch the worst case.
    //
    // Exception: if the new computed range is MUCH smaller than the cached
    // one (>= 10x), bust the cache. The cached value was inflated by an
    // earlier outlier (e.g. a torch.compile spike at step=0 polluting the
    // axis for the whole session). Without this, one bad iteration locks
    // a 20-second axis on a 30 Hz loop, making every normal bar render at
    // < 1 px width.
    const [newMin, newMax] = _computeStableRange(iterations);
    const _STALE_CACHE_RATIO = 10;
    const cachedSpan = state.ganttRange ? state.ganttRange[1] - state.ganttRange[0] : 0;
    const newSpan = newMax - newMin;
    if (state.ganttRange === null || (newSpan > 0 && cachedSpan > newSpan * _STALE_CACHE_RATIO)) {
        state.ganttRange = [newMin, newMax];
    } else {
        state.ganttRange = [Math.min(state.ganttRange[0], newMin), Math.max(state.ganttRange[1], newMax)];
    }

    _drawGantt(canvas, rec, state.ganttRange);
    _renderGanttLegend(scope, rec);
}

function _renderGanttLegend(scope, rec) {
    const legend = scope.querySelector('[data-role="gantt-legend"]');
    if (!legend) return;
    legend.innerHTML = '';
    const spans = rec.spans || {};
    const camEvents = rec.cam_events || {};
    // Span entries with matching swatch colors and per-stage durations so
    // the legend is also a quick numeric breakdown when a bar is too short
    // to label inside the canvas.
    for (const [name, [s, e]] of Object.entries(spans)) {
        const item = document.createElement('div');
        item.className = 'latency-gantt-legend-item';
        const color = _GANTT_SPAN_COLORS[name] || _GANTT_DEFAULT_SPAN_COLOR;
        item.innerHTML = `
            <span class="latency-gantt-legend-swatch" style="background:${color}"></span>
            <span>${name}</span>
            <span class="latency-gantt-legend-dur">${(e - s).toFixed(1)} ms</span>
        `;
        legend.appendChild(item);
    }
    // Cameras as a single legend row (they all share the diamond marker
    // color); per-camera staleness is already in the Cameras card row.
    const camKeys = Object.keys(camEvents);
    if (camKeys.length > 0) {
        const camRow = document.createElement('div');
        camRow.className = 'latency-gantt-legend-cam';
        camRow.innerHTML = `
            <span class="latency-gantt-legend-cam-marker">◆</span>
            <span>cameras (${camKeys.length})</span>
        `;
        legend.appendChild(camRow);
    }
}

function _drawGantt(canvas, rec, [minMs, maxMs]) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const spans = rec.spans || {};
    const camEvents = rec.cam_events || {};
    const loopDt = rec.loop_dt_ms || 0;

    const padL = 6, padR = 12, padT = 16, padB = 18;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xOf = ms => padL + plotW * (ms - minMs) / (maxMs - minMs);

    // Vertical zero line — separates background camera grab (left) from
    // in-iteration work (right). Subtle but important visually.
    const zeroX = xOf(0);
    ctx.strokeStyle = '#444';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(zeroX, padT - 2);
    ctx.lineTo(zeroX, H - padB + 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('iter_start', zeroX, padT - 4);

    // Loop-end line (right edge of the iteration's work).
    const endX = xOf(loopDt);
    ctx.strokeStyle = '#444';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(endX, padT - 2);
    ctx.lineTo(endX, H - padB + 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText('action_sent', endX, padT - 4);

    // X-axis: pick a "nice" tick interval based on the visible range, then
    // draw subtle vertical gridlines + ms labels at each tick. Lets the
    // operator read camera staleness (or any duration) directly off the
    // canvas instead of guessing from bar lengths.
    const range = maxMs - minMs;
    const tickStep = _niceTickStep(range, 8);  // aim for ~8 ticks max
    const firstTick = Math.ceil(minMs / tickStep) * tickStep;
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    // Edge-label width (corners always draw "{minMs} ms" / "{maxMs} ms").
    // Reserve a band around each corner so tick labels never collide
    // with the corner labels — that was the "-40 -40ms" overlap.
    const leftEdgeX = padL;
    const rightEdgeX = W - padR;
    const cornerLabelHalfWidth = 32;  // covers "-180 ms" at 10 px monospace
    for (let v = firstTick; v <= maxMs; v += tickStep) {
        const x = xOf(v);
        // Skip ticks too close to the iter_start / action_sent lines so
        // their labels don't visually collide with those distinguished ones.
        const tooCloseToZero = Math.abs(x - zeroX) < 14;
        const tooCloseToEnd = Math.abs(x - endX) < 14;
        // Skip ticks too close to either corner — the corner extent
        // labels always render and would visually merge with the tick.
        const tooCloseToLeft = (x - leftEdgeX) < cornerLabelHalfWidth;
        const tooCloseToRight = (rightEdgeX - x) < cornerLabelHalfWidth;
        if (!tooCloseToZero && !tooCloseToEnd && !tooCloseToLeft && !tooCloseToRight) {
            ctx.strokeStyle = '#1f1f3a';
            ctx.beginPath();
            ctx.moveTo(x, padT - 2);
            ctx.lineTo(x, H - padB);
            ctx.stroke();
            ctx.fillStyle = '#666';
            ctx.fillText(`${v}`, x, H - 4);
        }
    }
    // Always show the extent labels at the corners as a fallback for
    // narrow ranges where no tick happens to land near the edges.
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${minMs.toFixed(0)} ms`, padL, H - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxMs.toFixed(0)} ms`, W - padR, H - 4);

    // Reserve a fixed sliver at the bottom for the camera markers (one
    // sub-row per camera so labels don't stack on top of each other).
    const camKeys = Object.keys(camEvents);
    const camRowCount = Math.max(camKeys.length, 1);
    const camRowH = 12;
    const camAreaH = camKeys.length > 0 ? (camRowCount * camRowH + 4) : 0;

    // Lay out span bars in declaration order in the remaining vertical space.
    const spanNames = Object.keys(spans);
    const spanCount = Math.max(spanNames.length, 1);
    const spanRegionH = plotH - camAreaH;
    // Row height bumped from 20→30 max to give 10 px font + 4 px bar
    // margin comfortable headroom; otherwise labels clip on dense
    // iterations with 5+ spans plus 4 camera rows.
    const rowH = Math.min(30, Math.max(15, spanRegionH / spanCount));

    spanNames.forEach((name, i) => {
        const [s, e] = spans[name];
        const x0 = xOf(s);
        const x1 = xOf(e);
        const y = padT + i * rowH;
        const w = Math.max(2, x1 - x0);
        ctx.fillStyle = _GANTT_SPAN_COLORS[name] || _GANTT_DEFAULT_SPAN_COLOR;
        ctx.fillRect(x0, y, w, rowH - 4);
        // Label — name and duration if there's room. Drawn outside the bar
        // (to the right) when the bar is too narrow to hold any text, so
        // names are always readable.
        const dur = e - s;
        ctx.fillStyle = '#0a0a1a';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        const labelInside = w > 80
            ? `${name}  ${dur.toFixed(1)} ms`
            : (w > 36 ? name : '');
        if (labelInside) {
            ctx.fillText(labelInside, x0 + 3, y + rowH - 7);
        } else {
            // Place the label outside the bar to the right, in muted color.
            ctx.fillStyle = '#aaa';
            ctx.fillText(`${name} ${dur.toFixed(1)}ms`, x1 + 4, y + rowH - 7);
        }
    });

    // Camera markers — one sub-row per camera so the captured-→-consumed
    // line and the diamonds don't overlap with sibling cameras' markers.
    // Labels are intentionally NOT drawn here (per-camera staleness/period
    // is shown in the Cameras card row below the panel; duplicating it on
    // the Gantt was the source of overlapping text).
    const camAreaTop = H - padB - camAreaH + 2;
    camKeys.forEach((key, i) => {
        const ev = camEvents[key];
        const x = xOf(ev.captured_at_ms);
        const cx = xOf(ev.consumed_at_ms);
        const y = camAreaTop + i * camRowH + camRowH / 2;

        // Stale segment.
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(cx, y);
        ctx.stroke();

        // Capture diamond (filled).
        ctx.fillStyle = '#4fc3f7';
        ctx.beginPath();
        ctx.moveTo(x, y - 3); ctx.lineTo(x + 3, y); ctx.lineTo(x, y + 3); ctx.lineTo(x - 3, y); ctx.closePath();
        ctx.fill();
        // Consume diamond (hollow).
        ctx.strokeStyle = '#4fc3f7';
        ctx.beginPath();
        ctx.moveTo(cx, y - 2.5); ctx.lineTo(cx + 2.5, y); ctx.lineTo(cx, y + 2.5); ctx.lineTo(cx - 2.5, y); ctx.closePath();
        ctx.stroke();

        // Tiny camera label at the right edge so the row is identifiable.
        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(key, padL + plotW, y + 3);
    });
}

// Re-render the Gantt when the operator switches the percentile picker on
// any track. The track scope is the closest .latency-track ancestor; its
// state lives in _trackStates keyed by data-track-key.
document.addEventListener('change', e => {
    if (e.target && e.target.matches('[data-role="gantt-pick"]')) {
        const track = e.target.closest('.latency-track');
        if (!track) return;
        const state = _trackStates.get(track.dataset.trackKey);
        if (state) _renderGantt(track, state.latestIterations || {}, state);
    }
});

function _updateCameraOverlays(stages) {
    const overlays = document.querySelectorAll('.cam-latency-overlay');
    if (!overlays.length) return;
    for (const overlay of overlays) {
        const short = overlay.dataset.camShort;
        const stale = stages[`cam_${short}_stale_ms`];
        const period = stages[`cam_${short}_period_ms`];
        if (!stale) {
            overlay.style.display = 'none';
            continue;
        }
        const fps = (period && period.p50) ? (1000 / period.p50) : null;
        const stalePalette = stale.p95 < 40 ? 'ok' : stale.p95 < 70 ? 'warn' : 'bad';
        overlay.style.display = '';
        overlay.className = `cam-latency-overlay ${stalePalette}`;
        const fpsTxt = fps != null ? `${fps.toFixed(1)}&nbsp;Hz` : '—';
        overlay.innerHTML = `
            <div class="cam-latency-line">${fpsTxt}</div>
            <div class="cam-latency-line">stale&nbsp;${stale.p50.toFixed(0)}/${stale.p95.toFixed(0)}&nbsp;ms</div>
        `;
    }
}

function _startLatencyPoll() {
    _stopLatencyPoll();
    _showLatencyTab(true);
    // 1 Hz matches the snapshot writer interval; faster polling gains nothing.
    _latencyPollTimer = setInterval(_fetchLatencyMetrics, 1000);
    _fetchLatencyMetrics();
}

function _stopLatencyPoll() {
    if (_latencyPollTimer) { clearInterval(_latencyPollTimer); _latencyPollTimer = null; }
    // Reset per-track state so the next session starts fresh instead of
    // inheriting bounds / iteration caches from the previous run.
    _trackStates.clear();
    const container = document.getElementById('latency-tracks');
    if (container) {
        container.querySelectorAll('.latency-track').forEach(el => el.remove());
    }
}

async function _fetchLatencyMetrics() {
    // Discover available sources first; render only the fresh ones. Stale
    // sources (>5s old) are dropped from the display until they publish
    // again — this matches the polling cadence on the producer side.
    let sourcesData;
    try {
        const res = await fetch('/api/run/latency-sources');
        if (!res.ok) return;
        sourcesData = await res.json();
    } catch (e) { return; }
    const fresh = (sourcesData.sources || []).filter(s => s.fresh);
    _reconcileTracks(fresh);

    // Fetch each fresh source in parallel and render. Cameras live on the
    // robot, not on the inference thread; we collect cam_* stages from
    // every track and feed them to _updateCameraOverlays once so the
    // per-frame badges work regardless of which track owns the cameras.
    const results = await Promise.all(fresh.map(async source => {
        try {
            const res = await fetch(`/api/run/latency-metrics?source=${encodeURIComponent(source.key)}`);
            if (!res.ok) return null;
            return { source, data: await res.json() };
        } catch (e) { return null; }
    }));
    const mergedCamStages = {};
    for (const result of results) {
        if (!result) continue;
        const { source, data } = result;
        const trackEl = document.querySelector(`.latency-track[data-track-key="${CSS.escape(source.key)}"]`);
        if (!trackEl) continue;
        const state = _trackStates.get(source.key);
        if (!state) continue;
        _updateTrack(trackEl, data, state);
        for (const [k, v] of Object.entries(data.stages || {})) {
            if (k.startsWith('cam_')) mergedCamStages[k] = v;
        }
    }
    _updateCameraOverlays(mergedCamStages);
}

function _reconcileTracks(sources) {
    const container = document.getElementById('latency-tracks');
    if (!container) return;
    const noSources = document.getElementById('latency-no-sources');
    if (noSources) noSources.style.display = sources.length === 0 ? '' : 'none';

    // Remove tracks for sources that disappeared.
    const wantKeys = new Set(sources.map(s => s.key));
    container.querySelectorAll('.latency-track').forEach(el => {
        if (!wantKeys.has(el.dataset.trackKey)) {
            _trackStates.delete(el.dataset.trackKey);
            el.remove();
        }
    });

    // Add tracks that are new this poll.
    const have = new Set(
        [...container.querySelectorAll('.latency-track')].map(el => el.dataset.trackKey)
    );
    for (const source of sources) {
        if (have.has(source.key)) continue;
        container.appendChild(_makeTrackBlock(source));
        _trackStates.set(source.key, { latestIterations: {}, ganttRange: null });
    }
}

// Build a self-contained track DOM block. data-role attributes scope the
// Loop Health grid, Gantt selector / canvas / legend, camera grid, and
// meta footer so the rendering functions can find them with querySelector
// inside the track element instead of using global IDs.
function _makeTrackBlock(source) {
    const track = document.createElement('div');
    track.className = 'latency-track';
    track.dataset.trackKey = source.key;
    const display = source.loop_kind || source.key;
    track.innerHTML = `
        <div class="latency-track-header">${display}</div>
        <div class="latency-section">
            <div class="latency-section-title">Loop Health</div>
            <div class="latency-grid" data-role="grid-system"></div>
        </div>
        <div class="latency-section">
            <div class="latency-section-title">
                Iteration timeline
                <span class="latency-gantt-controls">
                    <select data-role="gantt-pick" title="Aggregate views synthesize a typical iteration where each stage independently shows its own percentile. Sample views show one real captured iteration whose per-stage values are whatever happened in that one iteration.">
                        <optgroup label="Aggregate (per-stage percentile)">
                            <option value="aggregate_median" selected>median</option>
                            <option value="aggregate_p95">p95</option>
                        </optgroup>
                        <optgroup label="Sample (real iteration)">
                            <option value="median">median sample</option>
                            <option value="p95">p95 sample</option>
                            <option value="p99">p99 sample</option>
                            <option value="max">worst (max loop)</option>
                            <option value="latest">latest</option>
                        </optgroup>
                    </select>
                    <span data-role="gantt-meta"></span>
                </span>
            </div>
            <div class="latency-gantt-panel" title="Per-iteration timeline. Bars are tracer spans; diamonds are camera frame captures (often before t=0 because cameras grab in a background thread). Gaps in the timeline indicate uninstrumented work; overlaps indicate parallel stages.">
                <div class="latency-gantt-canvas-wrap"><canvas data-role="gantt-canvas"></canvas></div>
                <div class="latency-gantt-legend" data-role="gantt-legend"></div>
                <div class="latency-empty" data-role="gantt-empty">No timeline yet — waiting for first snapshot…</div>
            </div>
        </div>
        <div class="latency-section">
            <div class="latency-section-title">Cameras</div>
            <div class="latency-cam-grid" data-role="cam-grid">
                <div class="latency-empty">No camera data — this track may not own cameras (e.g. inference thread).</div>
            </div>
        </div>
        <div class="latency-meta">
            <span data-role="meta-records">0 records</span>
            <span data-role="meta-dropped"></span>
            <span data-role="meta-updated"></span>
        </div>
    `;
    return track;
}

// Color thresholds for the system cards.
//
// Two of these (loop_dt_ms, get_observation_ms) are derived from the loop's
// target_period_ms (1000/fps), which the snapshot writer publishes — so
// they automatically scale to whatever FPS the teleop session is running.
// The rest are heuristic constants tuned for typical Feetech / 30 Hz
// camera setups; calibration drift detection (V2) will replace them.
//
// `ok`   → green: comfortably within budget.
// `warn` → yellow: approaching or just over the budget.
// `bad`  → red: clearly over budget; investigate.
function _budgetsFromTarget(targetPeriodMs) {
    // Fall back to 60 Hz when no target period is published (e.g. a session
    // that started before this field existed, or non-teleop loops).
    const t = targetPeriodMs || (1000 / 60);
    return {
        loop_dt_ms:         { ok: 0.7 * t, warn: t },         // 70% / 100% of FPS budget
        get_observation_ms: { ok: 0.4 * t, warn: 0.7 * t },   // obs read shouldn't dominate
        action_send_ms:     { ok: 2,       warn: 5 },         // sync_write is fire-and-forget
    };
}

// Per-card explanatory tooltips. Shown on hover via the standard `title`
// attribute. Keep terse — they should answer "what is this and why does
// the color change" in two sentences.
const _LAT_TOOLTIPS = {
    loop_dt_ms: 'Wall-clock time per iteration (work only; sleep excluded). The per-iteration budget is 1000 ms ÷ target FPS — at 60 fps that is 16.67 ms; at 30 fps, 33.33 ms. Color is driven by the p95 tail (the value on the right), not p50: a typical iteration may sit well under budget while the tail still hits overrun. Yellow at p95 ≥ 70% of budget; red when p95 exceeds it.',
    get_observation_ms: 'Cost of robot.get_observation(): the follower bus.sync_read("Present_Position") plus per-camera read_latest() calls. Cameras are cached by their grab thread so the camera reads are essentially free; this number is dominated by the motor sync_read (Feetech/Dynamixel round-trip + any retries).',
    action_send_ms: 'Cost of bus.sync_write("Goal_Position"). Fire-and-forget on Feetech, so this is bus-TX only — not motor motion.',
    overrun: 'Fraction of iterations whose work time exceeded the per-iteration budget (1000 ms ÷ target FPS). Sleep time is excluded — this only fires when the work itself is too slow to sustain the target rate.',
    cam_stale: 'Age of the cached camera frame at the moment the consumer reads it (now − cam.latest_timestamp). Yellow when p95 exceeds ~1.2× the camera frame period.',
    cam_period: 'Wall-clock interval between successive frame captures by the camera grab thread. Inverse is effective FPS.',
};

function _classifyLatency(value, budget) {
    if (budget == null) return '';
    if (value <= budget.ok) return 'ok';
    if (value <= budget.warn) return 'warn';
    return 'bad';
}

function _renderLatencyCard(parent, key, title, p50, p95, budget, sparkData, tooltip) {
    const card = document.createElement('div');
    card.className = 'latency-card';
    if (tooltip) card.title = tooltip;
    // Color reflects the TAIL (p95) — that's what determines whether the
    // loop is at risk of overrun under load, not the typical-case p50.
    // The headline value shows BOTH so the operator can see at a glance
    // which number is driving the color; the colored half is p95.
    const cls = _classifyLatency(p95 != null ? p95 : p50, budget);
    const p95Html = p95 != null
        ? ` <span class="latency-card-p95 ${cls}">p95 ${p95.toFixed(1)}</span>`
        : '';
    card.innerHTML = `
        <div class="latency-card-title">${title}</div>
        <div class="latency-card-value"><span class="latency-card-p50">p50 ${p50.toFixed(1)}</span>${p95Html}</div>
        <canvas data-spark-key="${key}"></canvas>
    `;
    parent.appendChild(card);
    if (sparkData && sparkData.length > 0) {
        const canvas = card.querySelector('canvas');
        canvas.id = `latency-spark-${key}`;
        // Reuse the same sparkline primitive RLT uses; auto-scale (no fixed range).
        drawChart(canvas.id, {series: [{data: sparkData, color: '#4fc3f7'}]});
    }
}

function _updateTrack(scope, data, state) {
    const stages = data.stages || {};
    const series = data.series || {};
    const targetPeriodMs = data.target_period_ms || null;
    const budgets = _budgetsFromTarget(targetPeriodMs);

    // Health badge — driven by snapshot.health.issues (populated by the
    // LoopHealthDetector on the producer side). Hidden when healthy; shows
    // an "error"-colored badge when any issue's severity is error, otherwise
    // a "warn"-colored one. Tooltip lists each issue's message so the
    // operator can drill in without leaving the panel.
    const issues = (data.health && data.health.issues) || [];
    const header = scope.querySelector('.latency-track-header');
    let badge = scope.querySelector('.latency-track-health');
    if (header) {
        if (issues.length === 0) {
            if (badge) badge.remove();
        } else {
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'latency-track-health';
                header.appendChild(badge);
            }
            const hasError = issues.some(i => i.severity === 'error');
            badge.classList.toggle('error', hasError);
            badge.classList.toggle('warn', !hasError);
            badge.textContent = `${issues.length} issue${issues.length > 1 ? 's' : ''}`;
            badge.title = issues.map(i => `[${i.severity}] ${i.message}`).join('\n');
        }
    }

    // System metrics row — sparkline cards for the stages we want to track
    // over time. Stages not in this set still appear in the Gantt below;
    // the Loop Health row is just the headline numbers per loop kind.
    const sysGrid = scope.querySelector('[data-role="grid-system"]');
    if (sysGrid) {
        sysGrid.innerHTML = '';
        // Top-row metrics are the LOAD-BEARING headline. Only show stats
        // that are at a comparable level of granularity:
        //   - loop_dt_ms: the umbrella (includes every stage, sleep
        //     excluded)
        //   - overrun_ratio: a derived rate, conceptually different from
        //     a duration but still a headline KPI
        // Per-stage breakdowns belong in the Gantt below, NOT here —
        // mixing them at top creates misleading impression that loop
        // ≈ obs + send (it doesn't; process_obs, process_action,
        // inference, dataset_write are also inside loop and would
        // distort the picture if absent from the headline).
        const sysOrder = [
            { key: 'loop_dt_ms', title: 'Loop' },
        ];
        for (const item of sysOrder) {
            const stage = stages[item.key];
            if (!stage) continue;
            const sparkPoints = (series[item.key] || []).map(([_t, v]) => v);
            _renderLatencyCard(sysGrid, item.key, item.title,
                stage.p50 ?? 0, stage.p95 ?? null,
                budgets[item.key], sparkPoints, _LAT_TOOLTIPS[item.key]);
        }

        // Overrun ratio.
        const overrun = (data.overrun_ratio || 0) * 100;
        const ovCard = document.createElement('div');
        ovCard.className = 'latency-card';
        ovCard.title = _LAT_TOOLTIPS.overrun + (
            targetPeriodMs ? ` Budget: ${targetPeriodMs.toFixed(2)} ms (${(1000 / targetPeriodMs).toFixed(0)} Hz target).` : ''
        );
        const ovCls = overrun < 1 ? 'ok' : overrun < 5 ? 'warn' : 'bad';
        const budgetLabel = targetPeriodMs
            ? `> ${targetPeriodMs.toFixed(1)} ms (${(1000 / targetPeriodMs).toFixed(0)} Hz)`
            : 'budget unknown';
        ovCard.innerHTML = `
            <div class="latency-card-title">Overrun</div>
            <div class="latency-card-value ${ovCls}">${overrun.toFixed(1)}%</div>
            <div class="latency-card-sub">Iterations with work time ${budgetLabel}.</div>
        `;
        sysGrid.appendChild(ovCard);
    }

    // Gantt timeline — pick one representative iteration and draw it.
    _renderGantt(scope, data.iterations || {}, state);

    // Per-camera cards.
    const camGrid = scope.querySelector('[data-role="cam-grid"]');
    if (camGrid) {
        const camKeys = Object.keys(stages).filter(k => k.startsWith('cam_') && k.endsWith('_stale_ms'));
        if (camKeys.length === 0) {
            camGrid.innerHTML = '<div class="latency-empty">No camera data — this track may not own cameras (e.g. inference thread).</div>';
        } else {
            camGrid.innerHTML = '';
            camKeys.sort();
            for (const staleKey of camKeys) {
                const camName = staleKey.slice('cam_'.length, -'_stale_ms'.length);
                const periodKey = `cam_${camName}_period_ms`;
                const stale = stages[staleKey];
                const period = stages[periodKey];
                const fps = period && period.p50 ? (1000 / period.p50) : null;
                // Stale threshold scales with the camera's actual period (when
                // we know it): yellow at 1.2× period, red at 2× period.
                const stalePeriod = period ? period.p50 : null;
                const okStale = stalePeriod ? stalePeriod * 1.2 : 40;
                const warnStale = stalePeriod ? stalePeriod * 2 : 70;
                const stalePalette = stale.p95 < okStale ? 'ok' : stale.p95 < warnStale ? 'warn' : 'bad';
                const card = document.createElement('div');
                card.className = 'latency-cam-card';
                card.title = `${_LAT_TOOLTIPS.cam_stale}\n\n${_LAT_TOOLTIPS.cam_period}`;
                card.innerHTML = `
                    <div class="latency-cam-card-title">${camName}</div>
                    <div class="latency-cam-stat">stale p50/p95 <b class="${stalePalette}">${stale.p50.toFixed(0)}/${stale.p95.toFixed(0)} ms</b></div>
                    <div class="latency-cam-stat">period p50 <b>${period ? period.p50.toFixed(1) : '—'} ms</b></div>
                    <div class="latency-cam-stat">effective fps <b>${fps != null ? fps.toFixed(1) : '—'} Hz</b></div>
                `;
                camGrid.appendChild(card);
            }
        }
    }

    // Footer.
    const recordsEl = scope.querySelector('[data-role="meta-records"]');
    if (recordsEl) recordsEl.textContent = `${data.n_records || 0} records`;
    const droppedEl = scope.querySelector('[data-role="meta-dropped"]');
    if (droppedEl) droppedEl.textContent = data.dropped_records ? `${data.dropped_records} dropped` : '';
    const updatedEl = scope.querySelector('[data-role="meta-updated"]');
    if (updatedEl) updatedEl.textContent = data.t ? `updated ${new Date(data.t * 1000).toLocaleTimeString()}` : '';
}

async function _fetchRLTMetrics() {
    try {
        const res = await fetch('/api/run/rlt-metrics');
        if (!res.ok) return;
        const data = await res.json();
        _updateRLTDashboard(data);
    } catch (e) { /* ignore */ }
}

function _updateRLTDashboard(data) {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('rlt-stat-episode', data.episode);
    set('rlt-stat-mode', data.mode);
    set('rlt-stat-rate', Math.round((data.autonomous_rate || 0) * 100) + '%');
    set('rlt-stat-throughput', (data.throughput_10min || 0) + '/10min');
    set('rlt-stat-buffer', data.buffer_size?.toLocaleString() || '0');
    set('rlt-stat-updates', data.total_updates?.toLocaleString() || '0');

    // Three independent groups, each with its own timestamps. The chart
    // for each group reads from its own group sub-dict; the hover-time
    // label per chart uses that group's timestamps. This is the layout
    // change that fixed the actor_deltas piggyback bug — actor delta is
    // a per-inference event, Q values / critic loss are per-grad-update,
    // and they tick at different rates.
    const series = data.series || {};
    const inf = series.inferences || {deltas: [], timestamps: []};
    const grad = series.grad_updates || {
        critic_losses: [], critic_grad_norms: [], actor_losses: [],
        q_values_mean: [], q_values_min: [], q_values_max: [],
        actor_q_terms: [], actor_bc_terms: [], update_rates: [], timestamps: [],
    };

    const nMax = parseInt(document.getElementById('rlt-stat-history')?.value) || 1000;
    const clip = arr => (arr && arr.length > nMax) ? arr.slice(-nMax) : (arr || []);

    // Success rate (episode-axis; standalone — not part of the synced group)
    if (series.autonomous_rate_rolling && series.autonomous_rate_rolling.length > 0) {
        drawChart('rlt-chart-success', {
            series: [{data: series.autonomous_rate_rolling, color: '#4fc3f7'}],
            fixedMin: 0, fixedMax: 1, percentage: true,
        });
    }

    const rltStep = data.total_updates || 0;

    // Per-grad-update synced charts (Q, critic loss, actor components, update rate).
    // All share the grad_updates timestamps.
    const gradTs = clip(grad.timestamps);
    if (grad.q_values_mean && grad.q_values_mean.length > 0) {
        drawChart('rlt-chart-qvalues', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.q_values_min), color: '#2ecc71', label: 'min',
             bandPair: 'min', bandColor: '#2ecc71', hideLine: true},
            {data: clip(grad.q_values_max), color: '#2ecc71', label: 'max',
             bandPair: 'max', hideLine: true},
            {data: clip(grad.q_values_mean), color: '#2ecc71', label: 'mean'},
        ]});
    }
    if (grad.critic_losses && grad.critic_losses.length > 0) {
        drawChart('rlt-chart-critic', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.critic_losses), color: '#f39c12', label: 'critic'},
        ]});
    }
    if ((grad.actor_q_terms && grad.actor_q_terms.length > 0) ||
        (grad.actor_bc_terms && grad.actor_bc_terms.length > 0)) {
        drawChart('rlt-chart-actor-components', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.actor_q_terms), color: '#e06c75', label: 'q'},
            {data: clip(grad.actor_bc_terms), color: '#4fc3f7', label: 'bc'},
        ]});
    }
    if (grad.update_rates && grad.update_rates.length > 0) {
        drawChart('rlt-chart-update-rate', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.update_rates), color: '#bd93f9', label: 'rate'},
        ]});
    }

    // Per-inference chart (actor delta). Its own timestamps — these
    // tick faster than grad_updates because inference fires every
    // query interval, while grad_updates fires only when the buffer
    // has reached the training threshold.
    if (inf.deltas && inf.deltas.length > 0) {
        drawChart('rlt-chart-delta', {syncGroup: 'rlt', latestStep: rltStep, timestamps: clip(inf.timestamps), series: [
            {data: clip(inf.deltas), color: '#e5c07b', label: 'delta'},
        ]});
    }
}

let _rltConfigTimer = null;
// Slider live-tuning controls only make sense during an active run —
// β / σ have no "preference" semantics, just a current value that the
// subprocess polls. Disable them visibly when no run is active.
// (The diag button is sticky — see _applyDiagBtnVisual.)
function _setRLTSlidersEnabled(enabled) {
    const tooltip = enabled ? '' : 'No active RLT training session';
    for (const id of ['rlt-slider-beta', 'rlt-slider-sigma']) {
        const el = document.getElementById(id);
        if (!el) continue;
        el.disabled = !enabled;
        el.title = tooltip;
        el.classList.toggle('disabled', !enabled);
    }
}

// Sticky-preference model for the Diagnostic toggle: the user's intent
// is stored in localStorage and survives across runs / page reloads.
// If toggled while no run is active, the click is staged locally and
// auto-applied at the start of the next run (see _initRLTSliders). If
// toggled mid-run, both the local pref and the backend are updated.
const _RLT_DIAG_PREF_KEY = 'rlt_dump_chunks_pref';

function _readDiagPref() {
    return localStorage.getItem(_RLT_DIAG_PREF_KEY) === '1';
}

function _writeDiagPref(on) {
    localStorage.setItem(_RLT_DIAG_PREF_KEY, on ? '1' : '0');
}

function _applyDiagBtnVisual(on, sessionActive) {
    const diagBtn = document.getElementById('rlt-btn-diagnostic');
    if (!diagBtn) return;
    diagBtn.classList.toggle('active', on);
    diagBtn.textContent = on ? 'Diagnostic: ON' : 'Diagnostic: OFF';
    diagBtn.title = sessionActive
        ? ''
        : `Will apply when training starts (currently ${on ? 'ON' : 'OFF'})`;
}

async function _initRLTSliders() {
    const controls = document.getElementById('rlt-controls');
    if (controls) controls.style.display = '';

    const betaSlider = document.getElementById('rlt-slider-beta');
    const sigmaSlider = document.getElementById('rlt-slider-sigma');
    const diagBtn = document.getElementById('rlt-btn-diagnostic');
    const betaVal = document.getElementById('rlt-slider-beta-val');
    const sigmaVal = document.getElementById('rlt-slider-sigma-val');

    // Seed controls from the backend so sliders show the values actually
    // applied to the training process, not the HTML's hardcoded defaults.
    // Without this, a page reload makes the UI lie about current β / σ.
    let sessionActive = false;
    let serverDiag = false;
    try {
        const resp = await fetch('/api/run/rlt-config');
        if (resp.ok) {
            const cfg = await resp.json();
            sessionActive = !!cfg.active;
            serverDiag = !!cfg.dump_chunks;
            if (betaSlider && typeof cfg.beta === 'number') {
                betaSlider.value = cfg.beta;
                if (betaVal) betaVal.textContent = cfg.beta.toFixed(2);
            }
            if (sigmaSlider && typeof cfg.exploration_sigma === 'number') {
                sigmaSlider.value = cfg.exploration_sigma;
                if (sigmaVal) sigmaVal.textContent = cfg.exploration_sigma.toFixed(3);
            }
        }
    } catch (e) { /* backend unreachable — keep HTML defaults */ }
    _setRLTSlidersEnabled(sessionActive);

    // Resolve diag state. Without a stored pref the button mirrors the
    // server (or HTML default OFF when no session). With a stored pref,
    // that's the user's intent and wins. If a session is active and the
    // pref differs from what the server is currently applying, sync the
    // pref to the backend now — this is how toggles staged before the
    // run actually take effect when the run starts.
    const havePref = localStorage.getItem(_RLT_DIAG_PREF_KEY) !== null;
    const desired = havePref ? _readDiagPref() : serverDiag;
    _applyDiagBtnVisual(desired, sessionActive);
    if (sessionActive && havePref && desired !== serverDiag) {
        try {
            await fetch('/api/run/rlt-config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dump_chunks: desired}),
            });
        } catch (e) { /* will retry on next click */ }
    }

    if (betaSlider) {
        betaSlider.oninput = () => {
            betaVal.textContent = parseFloat(betaSlider.value).toFixed(2);
            _sendRLTConfig();
        };
    }
    if (sigmaSlider) {
        sigmaSlider.oninput = () => {
            sigmaVal.textContent = parseFloat(sigmaSlider.value).toFixed(3);
            _sendRLTConfig();
        };
    }

    // Drop focus from RLT controls after release so the operator's arrow-key
    // taps go to the robot (← abort, → success, ↓ ignore — captured by the
    // subprocess's OS-level listener) instead of accidentally re-tweaking
    // whichever slider was last touched. Same problem class as the
    // diagnostic-button SPACE-retoggle bug.
    const _dropFocusAfterRelease = (el) => {
        if (!el) return;
        const blur = () => el.blur();
        el.addEventListener('mouseup', blur);
        el.addEventListener('touchend', blur);
        el.addEventListener('keyup', blur);
    };
    _dropFocusAfterRelease(betaSlider);
    _dropFocusAfterRelease(sigmaSlider);

    const historySelect = document.getElementById('rlt-stat-history');
    if (historySelect) {
        historySelect.onchange = () => _fetchRLTMetrics();
    }
    _dropFocusAfterRelease(historySelect);

    if (diagBtn) {
        diagBtn.onclick = async () => {
            // Drop focus so SPACE/ENTER don't keep re-triggering this button.
            // The subprocess captures SPACE for intervention; if this button
            // keeps focus after click, SPACE fires both the intervention and
            // this toggle, flipping the diagnostic flag on every intervention.
            diagBtn.blur();
            const on = !diagBtn.classList.contains('active');
            // Persist intent BEFORE the POST so a refresh / crash mid-flight
            // still preserves the user's choice. The POST is best-effort.
            _writeDiagPref(on);
            // Single POST. The server's 409 is the source of truth for
            // "no active session" — no separate probe GET to race with the
            // POST or to delay the click feedback. 409 → stage; 200 → live.
            let live = false;
            try {
                const resp = await fetch('/api/run/rlt-config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({dump_chunks: on}),
                });
                if (resp.status === 409) {
                    // No active session — preference is staged locally and
                    // will auto-apply at the start of the next run via
                    // _initRLTSliders. Silent (no toast) since this is the
                    // documented sticky-pref path, not an error.
                    live = false;
                } else if (resp.ok) {
                    live = true;
                } else {
                    const err = await resp.json().catch(() => ({}));
                    showToast('Diagnostic toggle failed',
                              err.detail || resp.statusText, 'error');
                }
            } catch (e) {
                showToast('Diagnostic toggle failed', e.message, 'error');
            }
            _applyDiagBtnVisual(on, live);
        };
    }
}

function _sendRLTConfig() {
    // Debounce: wait 300ms after last change before sending
    if (_rltConfigTimer) clearTimeout(_rltConfigTimer);
    _rltConfigTimer = setTimeout(async () => {
        const beta = parseFloat(document.getElementById('rlt-slider-beta')?.value);
        const sigma = parseFloat(document.getElementById('rlt-slider-sigma')?.value);
        if (isNaN(beta) || isNaN(sigma)) return;
        try {
            await fetch('/api/run/rlt-config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({beta, exploration_sigma: sigma}),
            });
        } catch (e) { /* ignore */ }
    }, 300);
}
