/* LeRobot GUI - Model Tab Logic */

let modelTabInitialized = false;
let modelSources = [];
let modelSourceData = {};  // sourcePath -> list of model entries
let expandedModelSources = new Set();
let expandedModelRuns = new Set();  // run paths with expanded checkpoint lists
let selectedModelRun = null;  // {path, name, ...}

// ============================================================================
// Initialization
// ============================================================================

async function modelTabInit() {
    if (modelTabInitialized) return;
    modelTabInitialized = true;
    await loadModelSources();
}

// ============================================================================
// Source management
// ============================================================================

async function loadModelSources() {
    try {
        const res = await fetch('/api/models/sources');
        if (!res.ok) return;
        modelSources = await res.json();
        expandedModelSources.clear();
        for (const s of modelSources) {
            if (s.expanded) expandedModelSources.add(s.path);
        }
        renderModelSources();
        const scanPromises = [];
        for (const s of modelSources) {
            if (s.expanded) scanPromises.push(scanModelSource(s.path));
        }
        await Promise.all(scanPromises);
    } catch (e) {
        console.error('Failed to load model sources:', e);
    }
}

async function scanModelSource(sourcePath) {
    const sid = _modelSourceId(sourcePath);
    const container = document.getElementById(`model-source-children-${sid}`);
    if (container) container.innerHTML = '<div class="source-loading">Scanning...</div>';
    try {
        const res = await fetch(`/api/models/sources/${encodeURIComponent(sourcePath)}/models`);
        if (!res.ok) throw new Error(await res.text());
        modelSourceData[sourcePath] = await res.json();
        renderModelSources();
    } catch (e) {
        console.error(`Failed to scan model source ${sourcePath}:`, e);
        if (container) container.innerHTML = '<div class="source-empty">Scan failed</div>';
    }
}

function _modelSourceId(path) {
    return 'ms_' + path.replace(/[^a-zA-Z0-9]/g, '_');
}

async function toggleModelSource(sourcePath) {
    if (expandedModelSources.has(sourcePath)) {
        expandedModelSources.delete(sourcePath);
    } else {
        expandedModelSources.add(sourcePath);
        if (!modelSourceData[sourcePath]) {
            scanModelSource(sourcePath);
        }
    }
    fetch(`/api/models/sources/${encodeURIComponent(sourcePath)}/expanded?expanded=${expandedModelSources.has(sourcePath)}`, { method: 'PUT' });
    renderModelSources();
}

async function addModelSource() {
    const path = prompt('Enter folder path to scan for training runs:');
    if (!path) return;
    try {
        const res = await fetch('/api/models/sources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        if (!res.ok) {
            const data = await res.json().catch(() => ({ detail: 'Failed to add source' }));
            throw new Error(data.detail || 'Failed to add source');
        }
        await loadModelSources();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function removeModelSource(sourcePath, e) {
    e.stopPropagation();
    if (!confirm(`Remove model source folder?\n${sourcePath}`)) return;
    try {
        const res = await fetch(`/api/models/sources/${encodeURIComponent(sourcePath)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to remove source');
        delete modelSourceData[sourcePath];
        expandedModelSources.delete(sourcePath);
        await loadModelSources();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

// ============================================================================
// Source tree rendering
// ============================================================================

function _esc(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function _formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(0) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

function _formatParams(n) {
    if (n == null) return '?';
    if (n < 1000) return n.toString();
    if (n < 1e6) return (n / 1e3).toFixed(1) + 'K';
    if (n < 1e9) return (n / 1e6).toFixed(1) + 'M';
    return (n / 1e9).toFixed(2) + 'B';
}

function _policyBadge(type) {
    if (!type) return '';
    const colors = {
        act: '#4fc3f7', diffusion: '#81c784', pi0: '#ffb74d', pi0_fast: '#ffb74d',
        pi05: '#ff8a65', vqbet: '#ce93d8', smolvla: '#a5d6a7',
    };
    const color = colors[type] || '#888';
    return `<span class="model-policy-badge" style="color:${color}">${_esc(type)}</span>`;
}

function renderModelSources() {
    const container = document.getElementById('model-sources-container');
    if (!container) return;

    if (modelSources.length === 0) {
        container.innerHTML = '<div class="source-empty">No sources configured</div>';
        return;
    }

    let html = '';
    for (const source of modelSources) {
        const isExpanded = expandedModelSources.has(source.path);
        const sid = _modelSourceId(source.path);
        const models = modelSourceData[source.path] || [];
        const countText = models.length > 0 ? `${models.length}` : '';
        const parts = source.path.split('/').filter(Boolean);
        const displayPath = parts.length > 2 ? '.../' + parts.slice(-2).join('/') : source.path;

        html += `<div class="source-folder">`;
        html += `<div class="source-folder-header" onclick="toggleModelSource('${source.path.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${source.path.replace(/'/g, "\\'")}')" title="${source.path}">`;
        html += `<span class="source-folder-toggle">${isExpanded ? '▼' : '▶'}</span>`;
        html += `<span class="source-folder-path">${displayPath}</span>`;
        html += `<span class="source-folder-count">${countText}</span>`;
        if (source.removable) {
            html += `<span class="source-folder-remove" onclick="removeModelSource('${source.path.replace(/'/g, "\\'")}', event)" title="Remove source">&times;</span>`;
        }
        html += `</div>`;

        html += `<div class="source-folder-children ${isExpanded ? 'expanded' : ''}" id="model-source-children-${sid}">`;
        if (isExpanded) {
            if (models.length === 0 && !modelSourceData[source.path]) {
                html += '<div class="source-loading">Scanning...</div>';
            } else if (models.length === 0) {
                html += '<div class="source-empty">No training runs found</div>';
            } else {
                for (const m of models) {
                    const isSelected = selectedModelRun && selectedModelRun.path === m.path;
                    const progress = (m.current_step && m.total_steps)
                        ? `${m.current_step.toLocaleString()}/${m.total_steps.toLocaleString()}`
                        : (m.current_step ? `${m.current_step.toLocaleString()} steps` : '');
                    html += `<div class="source-dataset${isSelected ? ' active' : ''}" onclick="selectModelRun('${m.path.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${m.path.replace(/'/g, "\\'")}', true)" title="${m.path}\n${m.policy_type} | ${progress}">`;
                    html += `<span class="source-dataset-name">${_esc(m.name)}</span>`;
                    html += `<span class="source-dataset-meta">${_esc(m.policy_type)}</span>`;
                    html += `</div>`;
                }
            }
        }
        html += `</div></div>`;
    }
    container.innerHTML = html;
}

// ============================================================================
// Model detail panel
// ============================================================================

async function selectModelRun(runPath) {
    // Find the run data from our cached source scans
    let runData = null;
    for (const models of Object.values(modelSourceData)) {
        runData = models.find(m => m.path === runPath);
        if (runData) break;
    }
    if (!runData) {
        showToast('Error', 'Model run not found', 'error');
        return;
    }

    selectedModelRun = runData;
    renderModelSources();  // Update selection highlight

    // Load checkpoints and config in parallel
    const [checkpoints, config] = await Promise.all([
        fetch(`/api/models/run/${encodeURIComponent(runPath)}/checkpoints`).then(r => r.ok ? r.json() : []),
        fetch(`/api/models/run/${encodeURIComponent(runPath)}/config`).then(r => r.ok ? r.json() : null),
    ]);

    renderModelDetail(runData, checkpoints, config);
}

function renderModelDetail(run, checkpoints, config) {
    const empty = document.getElementById('model-empty');
    const detail = document.getElementById('model-detail');
    if (!detail) return;

    // Model tab's right pane has three sibling containers — #model-empty,
    // #model-detail, #training-detail. Hiding the first two without also
    // hiding training-detail leaves the training form/run-detail visible
    // underneath the model header. Reset training state so its poll loop
    // won't re-render on top.
    if (typeof window.trainingLeaveView === 'function') window.trainingLeaveView();

    if (empty) empty.style.display = 'none';
    detail.style.display = '';

    let html = '';

    // Header
    html += `<div class="model-detail-header">`;
    html += `<h2>${_esc(run.name)}</h2>`;
    html += `<button class="btn-tiny" onclick="openModelFolder('${run.path.replace(/'/g, "\\'")}')">Open Folder</button>`;
    html += `<button class="btn-tiny btn-accent" onclick="testModelOnRobot('${run.path.replace(/'/g, "\\'")}')">Test on Robot</button>`;
    html += `</div>`;

    // Tabs
    html += `<div class="model-tabs">`;
    html += `<button class="model-tab active" data-mtab="overview" onclick="switchModelTab('overview')">Overview</button>`;
    html += `<button class="model-tab" data-mtab="config" onclick="switchModelTab('config')">Config</button>`;
    html += `<button class="model-tab" data-mtab="checkpoints" onclick="switchModelTab('checkpoints')">Checkpoints</button>`;
    html += `</div>`;

    // Overview subtab
    html += `<div class="model-tab-content active" id="mtab-overview">`;
    html += renderOverview(run, config);
    html += `</div>`;

    // Config subtab
    html += `<div class="model-tab-content" id="mtab-config">`;
    html += config ? renderConfig(config) : '<div class="model-no-config">No train_config.json found</div>';
    html += `</div>`;

    // Checkpoints subtab
    html += `<div class="model-tab-content" id="mtab-checkpoints">`;
    html += renderCheckpoints(checkpoints, run);
    html += `</div>`;

    detail.innerHTML = html;
}

function renderOverview(run, config) {
    let html = '<div class="model-overview">';

    // Info grid
    html += '<div class="model-info-grid">';

    html += _infoRow('Policy', _policyBadge(run.policy_type));
    html += _infoRow('Dataset', _esc(run.dataset || 'unknown'));

    if (run.current_step != null) {
        const progress = run.total_steps
            ? `${run.current_step.toLocaleString()} / ${run.total_steps.toLocaleString()} (${Math.round(100 * run.current_step / run.total_steps)}%)`
            : `${run.current_step.toLocaleString()} steps`;
        html += _infoRow('Training', progress);
    }

    if (run.batch_size) html += _infoRow('Batch size', run.batch_size);
    html += _infoRow('Parameters', _formatParams(run.num_parameters));
    html += _infoRow('Checkpoints', run.num_checkpoints);

    // Config-derived fields
    if (config) {
        const opt = config.optimizer || {};
        if (opt.lr) html += _infoRow('Learning rate', opt.lr);
        if (config.seed != null) html += _infoRow('Seed', config.seed);
        const peft = config.peft;
        if (peft) html += _infoRow('PEFT', `${peft.method_type || 'none'} (r=${peft.r || '?'})`);
    }

    // WandB link
    if (run.wandb_run_id && run.wandb_project) {
        const url = `https://wandb.ai/${run.wandb_project}/runs/${run.wandb_run_id}`;
        html += _infoRow('WandB', `<a href="${url}" target="_blank" class="model-link">${_esc(run.wandb_run_id)}</a>`);
    }

    html += '</div>';
    html += '</div>';
    return html;
}

function _infoRow(label, value) {
    return `<div class="model-info-row"><span class="model-info-label">${label}</span><span class="model-info-value">${value}</span></div>`;
}

function renderConfig(config) {
    let html = '<div class="model-config">';
    html += _renderConfigSection('Dataset', config.dataset);
    html += _renderConfigSection('Policy', config.policy);
    html += _renderConfigSection('Optimizer', config.optimizer);
    if (config.scheduler) html += _renderConfigSection('Scheduler', config.scheduler);
    html += _renderConfigSection('WandB', config.wandb);
    if (config.peft) html += _renderConfigSection('PEFT', config.peft);

    // Top-level scalar fields
    const skip = new Set(['dataset', 'policy', 'optimizer', 'scheduler', 'wandb', 'peft', 'env', 'eval']);
    const topLevel = {};
    for (const [k, v] of Object.entries(config)) {
        if (!skip.has(k) && v !== null && typeof v !== 'object') topLevel[k] = v;
    }
    if (Object.keys(topLevel).length > 0) {
        html += _renderConfigSection('General', topLevel);
    }

    html += '</div>';
    return html;
}

function _renderConfigSection(title, data) {
    if (!data || typeof data !== 'object') return '';
    let html = `<div class="model-config-section">`;
    html += `<div class="model-config-title" onclick="this.parentElement.classList.toggle('collapsed')">${_esc(title)}</div>`;
    html += `<div class="model-config-body">`;
    for (const [k, v] of Object.entries(data)) {
        if (v === null) continue;
        const display = typeof v === 'object' ? JSON.stringify(v) : String(v);
        html += `<div class="model-config-row">`;
        html += `<span class="model-config-key">${_esc(k)}</span>`;
        html += `<span class="model-config-val">${_esc(display)}</span>`;
        html += `</div>`;
    }
    html += `</div></div>`;
    return html;
}

function renderCheckpoints(checkpoints, run) {
    if (!checkpoints.length) return '<div class="model-no-config">No checkpoints found</div>';

    let html = '<div class="model-checkpoints">';
    html += '<table class="model-ckpt-table">';
    html += '<thead><tr><th>Step</th><th>Parameters</th><th>Resumable</th><th>Actions</th></tr></thead>';
    html += '<tbody>';

    for (const ckpt of checkpoints) {
        const stepText = ckpt.step != null ? ckpt.step.toLocaleString() : ckpt.name;
        const lastBadge = ckpt.is_last ? ' <span class="model-last-badge">latest</span>' : '';
        html += `<tr>`;
        html += `<td>${stepText}${lastBadge}</td>`;
        html += `<td>${_formatParams(ckpt.num_parameters)}</td>`;
        html += `<td>${ckpt.has_training_state ? 'Yes' : 'No'}</td>`;
        html += `<td class="model-ckpt-actions">`;
        html += `<button class="btn-tiny" onclick="openModelFolder('${ckpt.path.replace(/'/g, "\\'")}')">Open</button>`;
        html += `</td>`;
        html += `</tr>`;
    }

    html += '</tbody></table>';
    html += '</div>';
    return html;
}

// ============================================================================
// Model tab interactions
// ============================================================================

function switchModelTab(tabName) {
    document.querySelectorAll('.model-tab').forEach(b => b.classList.toggle('active', b.dataset.mtab === tabName));
    document.querySelectorAll('.model-tab-content').forEach(c => c.classList.remove('active'));
    const content = document.getElementById(`mtab-${tabName}`);
    if (content) content.classList.add('active');
}

async function openModelFolder(path) {
    try {
        const res = await fetch('/api/models/open-in-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Failed' }));
            showToast('Error', err.detail, 'error');
        }
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

function testModelOnRobot(runPath) {
    // Resolve the policy path via the server-emitted default_policy_path
    // on the matching modelSourceData entry — the layout-aware string.
    // Reconstructing it client-side (the old `runPath + '/checkpoints/...'`
    // form) silently broke for the docker recipe (`<run>/output/...`)
    // and for flat HVLA-S2-VLM layouts.
    const findEntry = () => {
        const data = (typeof modelSourceData !== 'undefined') ? modelSourceData : {};
        for (const arr of Object.values(data)) {
            const hit = arr.find(m => m.path === runPath);
            if (hit) return hit;
        }
        return null;
    };

    // Switch to Run tab → Policy workflow
    switchTab('run');
    selectWorkflow('policy');
    // Pre-select the checkpoint once both the selector + the cached
    // modelSourceData entry are available.
    const trySelect = () => {
        const sel = document.getElementById('run-policy-checkpoint');
        if (!sel) return false;
        const entry = findEntry();
        if (!entry?.default_policy_path) return false;
        const ckptPath = entry.default_policy_path;
        for (const opt of sel.options) {
            if (opt.value === ckptPath) {
                sel.value = ckptPath;
                _prefillPolicyFields(runPath);
                return true;
            }
        }
        return false;
    };
    // Model data might still be loading; retry briefly
    if (!trySelect()) {
        setTimeout(trySelect, 500);
        setTimeout(trySelect, 2000);
    }
}

function _prefillPolicyFields(runPath) {
    // Find the run data from the model tab's cached scans
    const data = (typeof modelSourceData !== 'undefined') ? modelSourceData : {};
    for (const models of Object.values(data)) {
        const m = models.find(m => m.path === runPath);
        if (!m) continue;
        // Ensure dataset selector is on "New dataset" and show the name input
        const datasetSel = document.getElementById('run-policy-dataset');
        if (datasetSel) { datasetSel.value = '__new__'; }
        if (typeof _onPolicyDatasetChange === 'function') _onPolicyDatasetChange();
        // Pre-fill dataset repo_id with eval_ prefix
        const repoInput = document.getElementById('run-policy-repo-id');
        if (repoInput) {
            const runName = m.name.split('/').pop() || 'policy';
            repoInput.value = `eval/eval_${runName}`;
        }
        // Store training dataset info and update task selector
        if (typeof _policyTrainingDataset !== 'undefined') {
            _policyTrainingDataset = m.dataset
                ? { repo_id: m.dataset, root: m.dataset_root || null }
                : null;
        }
        if (typeof _updatePolicyTaskSelector === 'function') {
            _updatePolicyTaskSelector();
        }
        break;
    }
}
