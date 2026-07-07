/* LeRobot Dataset GUI - Application Logic */

let datasets = {};
let episodes = {};
// Bridge highlight state — repo_id → Set<episode_id>. Wired by
// bridge_consumers.js when the AI calls highlight_in_viewer; renderTree()
// reads from it so the cyan outline survives re-renders. Tab-scope (lost on
// reload), intentionally not persisted.
const bridgeHighlights = new Map();
window.bridgeHighlights = bridgeHighlights;
let expandedNodes = new Set();
let currentDataset = null;
let currentEpisode = null;
let currentFrame = 0;
let totalFrames = 0;
let isPlaying = false;
let playInterval = null;
let fps = 30;
let playbackSpeed = 1;
let isDragging = false;

// Editing state
let pendingEdits = [];
let contextMenuTarget = null;  // {datasetId, episodeIndex}

// Trim state
let trimStart = 0;  // Frame index
let trimEnd = 0;    // Frame index (exclusive, like end_frame in API)
let isDraggingTrimLeft = false;
let isDraggingTrimRight = false;
let justFinishedTrimDrag = false;

// Dataset sources (folder browser)
let sources = [];
let sourceDatasets = {};  // {sourcePath: [{name, root, total_episodes, ...}]}
let expandedSources = new Set();
let _sourcesLoaded = false;

// `let` at script-scope is NOT visible on `window` — sibling scripts
// (feature_editing.js, etc.) can read these via bare names but not via
// `window.X`. Mirror the shared state via getters so cross-file readers
// work either way. Read-only — sibling scripts must not assign these.
Object.defineProperties(window, {
    datasets: { get: () => datasets, configurable: true },
    episodes: { get: () => episodes, configurable: true },
    currentDataset: { get: () => currentDataset, configurable: true },
    currentEpisode: { get: () => currentEpisode, configurable: true },
    currentFrame: { get: () => currentFrame, configurable: true },
    totalFrames: { get: () => totalFrames, configurable: true },
    fps: { get: () => fps, configurable: true },
    trimStart: { get: () => trimStart, configurable: true },
    trimEnd: { get: () => trimEnd, configurable: true },
    pendingEdits: { get: () => pendingEdits, configurable: true },
});

async function loadSources() {
    try {
        const res = await fetch('/api/datasets/sources');
        if (!res.ok) return;
        sources = await res.json();
        // Restore expansion state
        expandedSources.clear();
        for (const s of sources) {
            if (s.expanded) expandedSources.add(s.path);
        }
        renderSources();
        // Scan expanded sources
        for (const s of sources) {
            if (s.expanded) {
                scanSource(s.path);
            }
        }
        _sourcesLoaded = true;
    } catch (e) {
        console.error('Failed to load sources:', e);
    }
}

async function scanSource(sourcePath) {
    const container = document.getElementById(`source-children-${_sourceId(sourcePath)}`);
    if (container) container.innerHTML = '<div class="source-loading">Scanning...</div>';
    try {
        const res = await fetch(`/api/datasets/sources/${encodeURIComponent(sourcePath)}/datasets`);
        if (!res.ok) throw new Error(await res.text());
        sourceDatasets[sourcePath] = await res.json();
        renderSources();
    } catch (e) {
        console.error(`Failed to scan source ${sourcePath}:`, e);
        if (container) container.innerHTML = '<div class="source-empty">Scan failed</div>';
    }
}

function _sourceId(path) {
    // Create a safe DOM id from a path
    return path.replace(/[^a-zA-Z0-9]/g, '_');
}

async function toggleSource(sourcePath) {
    if (expandedSources.has(sourcePath)) {
        expandedSources.delete(sourcePath);
    } else {
        expandedSources.add(sourcePath);
        // Scan if not yet loaded
        if (!sourceDatasets[sourcePath]) {
            scanSource(sourcePath);
        }
    }
    // Persist expansion state
    fetch(`/api/datasets/sources/${encodeURIComponent(sourcePath)}/expanded?expanded=${expandedSources.has(sourcePath)}`, { method: 'PUT' });
    renderSources();
}

async function addSource() {
    const path = prompt('Enter folder path to scan for datasets:');
    if (!path) return;
    try {
        const res = await fetch('/api/datasets/sources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        if (!res.ok) {
            const data = await res.json().catch(() => ({ detail: 'Failed to add source' }));
            throw new Error(data.detail || 'Failed to add source');
        }
        await loadSources();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function removeSource(sourcePath, e) {
    e.stopPropagation();
    if (!confirm(`Remove source folder?\n${sourcePath}`)) return;
    try {
        const res = await fetch(`/api/datasets/sources/${encodeURIComponent(sourcePath)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to remove source');
        delete sourceDatasets[sourcePath];
        expandedSources.delete(sourcePath);
        await loadSources();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

function openDatasetFromSource(root) {
    openDataset(root);
}

function renderSources() {
    const container = document.getElementById('sources-container');
    if (!container) return;

    if (sources.length === 0) {
        container.innerHTML = '<div class="source-empty">No sources configured</div>';
        return;
    }

    let html = '';
    for (const source of sources) {
        const isExpanded = expandedSources.has(source.path);
        const sid = _sourceId(source.path);
        const datasets = sourceDatasets[source.path] || [];
        const countText = datasets.length > 0 ? `${datasets.length}` : '';
        // Show last two path segments for readability
        const parts = source.path.split('/').filter(Boolean);
        const displayPath = parts.length > 2 ? '.../' + parts.slice(-2).join('/') : source.path;

        html += `<div class="source-folder">`;
        html += `<div class="source-folder-header" onclick="toggleSource('${source.path.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${source.path.replace(/'/g, "\\'")}')" title="${source.path}">`;
        html += `<span class="source-folder-toggle">${isExpanded ? '▼' : '▶'}</span>`;
        html += `<span class="source-folder-path">${displayPath}</span>`;
        html += `<span class="source-folder-count">${countText}</span>`;
        if (source.removable) {
            html += `<span class="source-folder-remove" onclick="removeSource('${source.path.replace(/'/g, "\\'")}', event)" title="Remove source">&times;</span>`;
        }
        html += `</div>`;

        html += `<div class="source-folder-children ${isExpanded ? 'expanded' : ''}" id="source-children-${sid}">`;
        if (isExpanded) {
            if (datasets.length === 0 && !sourceDatasets[source.path]) {
                html += '<div class="source-loading">Scanning...</div>';
            } else if (datasets.length === 0) {
                html += '<div class="source-empty">No datasets found</div>';
            } else {
                for (const ds of datasets) {
                    const isOpen = Object.keys(window.datasets || {}).some(id => {
                        const d = window.datasets[id];
                        return d && d.root === ds.root;
                    });
                    html += `<div class="source-dataset${isOpen ? ' active' : ''}" onclick="openDatasetFromSource('${ds.root.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${ds.root.replace(/'/g, "\\'")}')" title="${ds.root}\n${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames">`;
                    html += `<span class="source-dataset-name">${ds.name}</span>`;
                    html += `<span class="source-dataset-meta">${ds.total_episodes} ep</span>`;
                    html += `</div>`;
                }
            }
        }
        html += `</div></div>`;
    }
    container.innerHTML = html;
}

async function openDataset(path) {
    if (!path) return;

    setStatus('Opening dataset...');
    try {
        const body = path.startsWith('/') ? { local_path: path } : { repo_id: path };
        const res = await fetch('/api/datasets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        // 409 = incomplete local cache. Hand off to the Hub modal in
        // 'open-sync' mode — the modal owns the download + open flow from
        // here. We bail out of openDataset; the modal calls _completeOpen()
        // on success.
        if (res.status === 409) {
            const payload = await res.json();
            const detail = payload && payload.detail;
            if (detail && detail.code === 'incomplete_local_cache') {
                openHubModal(null, 'open-sync', { body, detail });
                return;
            }
        }

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        await _completeOpen(data);
    } catch (e) {
        let errorMsg = e.message;
        try {
            const parsed = JSON.parse(errorMsg);
            if (parsed.detail) errorMsg = parsed.detail;
        } catch (_) {}
        setStatus('Error: ' + errorMsg);
        showToast('Failed to open dataset', errorMsg, 'error', 10000);
    }
}

// Shared post-open flow: surface errors/warnings, load episodes, expand the
// tree, refresh edits. Called from both the normal openDataset path and the
// Hub-modal 'open-sync' path.
async function _completeOpen(data) {
    datasets[data.id] = data;

    if (data.errors && data.errors.length > 0) {
        showToast('Dataset Error', data.errors.join('\n'), 'error', 0);
    }
    if (data.warnings && data.warnings.length > 0) {
        const actionable = data.warnings.filter(w => !w.startsWith('stats.json mismatch'));
        if (actionable.length > 0) {
            showToast('Dataset Warning', actionable.join('\n'), 'warning', 8000);
        }
    }

    const epRes = await fetch(`/api/datasets/${encodeURIComponent(data.id)}/episodes`);
    episodes[data.id] = await epRes.json();
    datasets[data.id].total_episodes = episodes[data.id].length;

    expandedNodes.add(data.id);
    await refreshPendingEdits();

    renderTree();
    renderSources();
    if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
    if (window.FeatureEditing) window.FeatureEditing.onDatasetOpened(data.id);
    setStatus(`Opened: ${data.repo_id}`);
}

function isEpisodeDeleted(datasetId, epIdx) {
    return pendingEdits.some(e => e.dataset_id === datasetId && e.episode_index === epIdx && e.edit_type === 'delete');
}

function isEpisodeTrimmed(datasetId, epIdx) {
    return pendingEdits.some(e => e.dataset_id === datasetId && e.episode_index === epIdx && e.edit_type === 'trim');
}

// Derive episode quality flags from the per-component action stats the
// API exposes (min, max, mean, std arrays — pre-computed at record time).
// Backend just surfaces the raw characteristics; consumers decide what
// counts as a problem. New checks can land here without touching the API.
function _episodeActionFlags(stats) {
    if (!stats) return { allZero: false };  // unknown — render no badges
    const absMaxOfMin = Math.max(...stats.min.map(Math.abs));
    const absMaxOfMax = Math.max(...stats.max.map(Math.abs));
    return {
        allZero: absMaxOfMin === 0 && absMaxOfMax === 0,
        // Future: static = Math.max(...stats.std) === 0
        // Future: jittery = mean(stats.std) > some_threshold
    };
}

function renderTree() {
    const container = document.getElementById('tree-container');
    if (Object.keys(datasets).length === 0) {
        container.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No datasets opened</div>';
        return;
    }

    let html = '';
    for (const [id, ds] of Object.entries(datasets)) {
        const isExpanded = expandedNodes.has(id);
        const dsEpisodes = episodes[id] || [];
        const dsEditCount = pendingEdits.filter(e => e.dataset_id === id).length;
        const totalFrames = dsEpisodes.reduce((sum, ep) => sum + ep.length, 0);
        const tooltip = `${ds.repo_id}\n${ds.total_episodes} episodes, ${totalFrames.toLocaleString()} frames\nPath: ${ds.root}`;

        html += `
            <div class="tree-node">
                <div class="tree-header" onclick="toggleDataset('${id}')" oncontextmenu="showFolderContextMenu(event, '${ds.root.replace(/'/g, "\\'")}')" title="${tooltip}">
                    <span class="tree-toggle">${isExpanded ? '▼' : '▶'}</span>
                    <span class="tree-icon">${ds.errors && ds.errors.length > 0 ? '⚠️' : '📁'}</span>
                    <span class="tree-label">${ds.repo_id}</span>
                    <span class="tree-meta">${dsEditCount > 0 ? `${dsEditCount}✎ ` : ''}${ds.total_episodes} ep</span>
                    <span class="tree-close" onclick="closeDataset('${id}', event)" title="Close">&times;</span>
                </div>
                <div class="tree-children ${isExpanded ? 'expanded' : ''}">
        `;

        for (const ep of dsEpisodes) {
            const isActive = currentDataset === id && currentEpisode === ep.episode_index;
            const isDeleted = isEpisodeDeleted(id, ep.episode_index);
            const isTrimmed = isEpisodeTrimmed(id, ep.episode_index);
            const hasVideoMismatch = ep.video_extra_frames !== 0;
            // Derive action-quality flags from the raw per-component stats
            // exposed by the API. New checks (static, saturated, jittery)
            // can be added here without touching the backend — the API just
            // surfaces the raw characteristics.
            const actionFlags = _episodeActionFlags(ep.action_stats);
            const hasZeroActions = actionFlags.allZero;
            // "quality-warning" is the unified visual state for any per-episode
            // quality issue. The tooltip distinguishes the cause.
            const hasQualityWarning = hasVideoMismatch || hasZeroActions;
            const isBridgeHighlight = bridgeHighlights.get(id)?.has(ep.episode_index);
            const classes = ['tree-header'];
            if (isActive) classes.push('active');
            if (isDeleted) classes.push('deleted');
            if (isTrimmed) classes.push('trimmed');
            if (hasQualityWarning) classes.push('quality-warning');
            if (isBridgeHighlight) classes.push('bridge-highlight');

            let icon = '🎬';
            if (isDeleted) icon = '🗑️';
            else if (hasQualityWarning) icon = '⚠️';

            let meta = `${ep.length} frames`;
            if (hasVideoMismatch) {
                const sign = ep.video_extra_frames > 0 ? '+' : '';
                meta += ` (${sign}${ep.video_extra_frames})`;
            }
            if (hasZeroActions) meta += ' (zero actions)';

            // Compose tooltip across all warnings on this episode.
            const tipParts = [];
            if (hasVideoMismatch) {
                tipParts.push(
                    ep.video_extra_frames > 0
                        ? `Video-data mismatch: ${ep.video_extra_frames} extra frames (re-recording artifact)`
                        : `Video-data mismatch: ${Math.abs(ep.video_extra_frames)} missing frames (truncated video)`
                );
            }
            if (hasZeroActions) {
                tipParts.push(
                    'Action column is identically zero across every frame — almost always a recording-flow bug ' +
                    '(intervention flag never engaged during teleop). Episode is useless for training/replay.'
                );
            }
            const titleAttr = tipParts.length ? `title="${tipParts.join('\n\n').replace(/"/g, '&quot;')}"` : '';

            html += `
                <div class="${classes.join(' ')}"
                     data-episode-row
                     data-dataset-id="${id}"
                     data-episode-id="${ep.episode_index}"
                     onclick="selectEpisode('${id}', ${ep.episode_index}, ${ep.video_length || ep.length})"
                     oncontextmenu="showContextMenu(event, '${id}', ${ep.episode_index})"
                     ${titleAttr}>
                    <span class="tree-toggle"></span>
                    <span class="tree-icon">${icon}</span>
                    <span class="tree-label">Episode ${ep.episode_index}</span>
                    <span class="tree-meta">${meta}</span>
                </div>
            `;
        }

        html += '</div></div>';
    }
    container.innerHTML = html;
    updateEditsBar();
}

function toggleDataset(id) {
    if (expandedNodes.has(id)) {
        expandedNodes.delete(id);
    } else {
        expandedNodes.add(id);
    }
    renderTree();
}

async function closeDataset(id, e) {
    e.stopPropagation();
    try {
        await fetch(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
        delete datasets[id];
        delete episodes[id];
        expandedNodes.delete(id);
        if (currentDataset === id) {
            currentDataset = null;
            currentEpisode = null;
            document.getElementById('camera-grid').innerHTML = '<div class="empty-state">Select an episode to view</div>';
        }
        renderTree();
        if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
        renderSources();
    } catch (err) {
        showToast('Error', 'Failed to close dataset: ' + err.message, 'error');
    }
}

function selectEpisode(datasetId, epIdx, length) {
    currentDataset = datasetId;
    currentEpisode = epIdx;
    totalFrames = length;
    currentFrame = 0;
    fps = datasets[datasetId].fps || 30;

    // Initialize trim to full episode
    trimStart = 0;
    trimEnd = totalFrames;

    // Stop playback
    if (isPlaying) {
        togglePlay();
    }

    renderTree();
    renderCameraGrid();
    loadAllFrames(0);
    loadTrimForCurrentEpisode();
    if (window.FeatureEditing) window.FeatureEditing.onEpisodeSelected(datasetId, epIdx);
}

function renderCameraGrid() {
    const grid = document.getElementById('camera-grid');
    if (!currentDataset || currentEpisode === null) {
        grid.innerHTML = '<div class="empty-state">Select an episode to view</div>';
        return;
    }

    const ds = datasets[currentDataset];
    const cameras = ds.camera_keys;
    // The URDF tile counts as one cell in the grid; treat it as a virtual
    // camera for layout purposes (and append it physically below). Whether
    // it survives is decided async by _probeAndAttachUrdfViz, which removes
    // the placeholder if this dataset's motor set has no vendored URDF.
    const tileCount = cameras.length + 1;

    let cols = 1;
    if (tileCount === 2) cols = 2;
    else if (tileCount >= 3 && tileCount <= 4) cols = 2;
    else if (tileCount >= 5) cols = 3;

    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    let html = '';
    for (const cam of cameras) {
        const camName = cam.split('.').pop();
        html += `
            <div class="camera-panel">
                <div class="camera-title">${camName}</div>
                <div class="camera-frame">
                    <img id="frame-${cam.replace(/\./g, '-')}" src="" alt="${camName}">
                </div>
            </div>
        `;
    }
    html += `
        <div class="camera-panel" id="urdf-viz-panel" style="display: none;">
            <div class="camera-title">visualizer</div>
            <div class="camera-frame">
                <iframe id="urdf-viz-iframe" src="" title="URDF state visualization"
                        style="width: 100%; height: 100%; border: none; background: #1a1a1a;"></iframe>
            </div>
        </div>
    `;
    grid.innerHTML = html;
    _probeAndAttachUrdfViz(currentDataset, currentEpisode);
}

let _urdfVizAvailability = {};  // dataset_id -> bool (cached after first probe)

// Per-tab persisted preference for the data-tab URDF ghost / trajectory
// toggle. Backed by sessionStorage so it survives episode changes (each
// selectEpisode rebuilds the camera grid via ``grid.innerHTML``, which
// destroys + recreates the iframe and loses its module-level ``_ghostOn``).
// Falls back to the parent's ``?urdfGhost=on`` URL param (the bookmarkable
// initial state, also what the screenshot script keys off).
function _urdfGhostPref() {
    const stored = sessionStorage.getItem('urdfGhost');
    if (stored !== null) return stored === 'on';
    return new URLSearchParams(location.search).get('urdfGhost') === 'on';
}

// One-time install: iframe postMessages ``urdfGhostChanged`` when the
// user clicks the toggle inside it. We update sessionStorage so the
// next iframe (built by the next selectEpisode) initializes with the
// remembered value via _urdfGhostPref above.
(function _wireUrdfGhostPersistence() {
    window.addEventListener('message', (ev) => {
        if (ev.data && ev.data.type === 'urdfGhostChanged') {
            sessionStorage.setItem('urdfGhost', ev.data.on ? 'on' : 'off');
        }
    });
})();

async function _probeAndAttachUrdfViz(datasetId, episodeIdx) {
    const panel = document.getElementById('urdf-viz-panel');
    const iframe = document.getElementById('urdf-viz-iframe');
    if (!panel || !iframe) return;

    let available = _urdfVizAvailability[datasetId];
    if (available === undefined) {
        try {
            const url = `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIdx}/urdf-viz?frame=0`;
            const r = await fetch(url);
            const d = await r.json();
            available = !!d.available;
        } catch (e) {
            available = false;
        }
        _urdfVizAvailability[datasetId] = available;
    }
    // Bail if the user has navigated away while we were probing — a later
    // selectEpisode call has re-rendered the grid and a new probe is in
    // flight for the new episode.
    if (currentDataset !== datasetId || currentEpisode !== episodeIdx) return;
    if (!available) {
        panel.remove();
        // Drop the empty cell back out of the column count.
        const grid = document.getElementById('camera-grid');
        const cams = datasets[datasetId].camera_keys.length;
        let cols = 1;
        if (cams === 2) cols = 2;
        else if (cams >= 3 && cams <= 4) cols = 2;
        else if (cams >= 5) cols = 3;
        grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        return;
    }
    panel.style.display = '';
    // mode=dataset means the iframe waits for postMessage frame updates from
    // the parent (this page), driven by the scrubber via _postFrameToUrdfViz.
    // ``_urdfGhostPref()`` reads sessionStorage first (sticky across episode
    // changes within a tab) then falls back to the parent URL's
    // ``?urdfGhost=on`` (bookmarkable initial state, used by the screenshot
    // script). Bump the version any time this seams (URL param contract or
    // postMessage protocol) changes so an old cached iframe doesn't stick.
    const ghostInit = _urdfGhostPref() ? '&ghost=on' : '';
    iframe.src = `/static/urdf_viz.html?mode=dataset&v=2${ghostInit}`;
    // Fast path: iframe.onload fires when the document is parsed, which is
    // usually before the module script has registered its message listener
    // but in practice fast enough for an idle main thread. Belt:
    // `urdfVizReady` from the iframe's module script (see urdf_viz.html)
    // arrives once the listener IS registered. Heavy main-thread load
    // (Playwright video recording, dev tools, etc.) can race the fast path;
    // the ready signal re-posts after the listener is guaranteed live.
    iframe.addEventListener('load', () => _postFrameToUrdfViz(currentFrame), { once: true });
}

// Re-post the frame on the iframe's ready signal — robust to the
// module-script-deferred-registration race the fast iframe.onload path
// can hit under heavy main-thread load.
window.addEventListener('message', (e) => {
    const msg = e.data;
    if (!msg || typeof msg !== 'object' || msg.type !== 'urdfVizReady') return;
    if (currentDataset == null || currentEpisode == null) return;
    _postFrameToUrdfViz(currentFrame);
});

function _postFrameToUrdfViz(frameIdx) {
    const iframe = document.getElementById('urdf-viz-iframe');
    if (!iframe || !iframe.contentWindow || !currentDataset || currentEpisode === null) return;
    iframe.contentWindow.postMessage(
        { type: 'frame', dataset: currentDataset, episode: currentEpisode, frame: frameIdx },
        '*',
    );
}

function loadAllFrames(idx) {
    if (!currentDataset || currentEpisode === null) return Promise.resolve();
    currentFrame = Math.max(0, Math.min(idx, totalFrames - 1));

    const ds = datasets[currentDataset];
    const promises = [];

    for (const cam of ds.camera_keys) {
        const url = `/api/datasets/${encodeURIComponent(currentDataset)}/episodes/${currentEpisode}/frame/${currentFrame}?camera=${encodeURIComponent(cam)}`;
        const imgId = `frame-${cam.replace(/\./g, '-')}`;
        const img = document.getElementById(imgId);
        if (img) {
            const promise = new Promise((resolve) => {
                img.onload = resolve;
                img.onerror = resolve; // Don't block on errors
            });
            img.src = url;
            promises.push(promise);
        }
    }

    // Update UI
    document.getElementById('frame-info').textContent = `${currentFrame + 1} / ${totalFrames}`;
    const pct = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0;
    document.getElementById('timeline-progress').style.width = `${pct}%`;
    document.getElementById('timeline-scrubber').style.left = `${pct}%`;

    // Update time display
    const currentTime = formatTime(currentFrame / fps);
    const totalTime = formatTime(totalFrames / fps);
    document.getElementById('time-info').textContent = `${currentTime} / ${totalTime}`;

    if (window.FeatureEditing) window.FeatureEditing.onPlayheadChanged();
    _postFrameToUrdfViz(currentFrame);

    return Promise.all(promises);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function playLoop() {
    while (isPlaying) {
        const frameTime = 1000 / (fps * playbackSpeed);
        const startTime = performance.now();

        // Constrain playback to trim region
        const playStart = trimStart;
        const playEnd = trimEnd - 1;  // trimEnd is exclusive

        if (currentFrame >= playEnd) {
            currentFrame = playStart;
        } else if (currentFrame < playStart) {
            currentFrame = playStart;
        } else {
            currentFrame++;
        }

        await loadAllFrames(currentFrame);

        // Wait remaining time to maintain fps (if frames loaded fast enough)
        const elapsed = performance.now() - startTime;
        const sleepTime = frameTime - elapsed;
        if (sleepTime > 0) {
            await new Promise(r => setTimeout(r, sleepTime));
        }
    }
}

function changeSpeed(speed) {
    playbackSpeed = parseFloat(speed);
}

function togglePlay() {
    if (!currentDataset || currentEpisode === null) return;

    isPlaying = !isPlaying;
    document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';

    if (isPlaying) {
        playLoop();
    }
}

async function launchRerun() {
    if (!currentDataset || currentEpisode === null) {
        setStatus('Select an episode first');
        return;
    }

    setStatus('Launching Rerun...');
    try {
        const res = await fetch(
            `/api/datasets/${encodeURIComponent(currentDataset)}/episodes/${currentEpisode}/visualize`,
            { method: 'POST' }
        );
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setStatus(data.message);
    } catch (e) {
        setStatus('Error: ' + e.message);
    }
}

function getFrameFromTimelineEvent(e) {
    const timeline = document.getElementById('timeline');
    const rect = timeline.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    return Math.floor(pct * (totalFrames - 1));
}

function seekTimeline(frame) {
    if (!currentDataset || currentEpisode === null) return;
    loadAllFrames(frame);
}

function updateHoverPreview(e) {
    if (!currentDataset || currentEpisode === null) return;
    const frame = getFrameFromTimelineEvent(e);
    const time = formatTime(frame / fps);
    const hover = document.getElementById('timeline-hover');
    const timeline = document.getElementById('timeline');
    const rect = timeline.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    hover.style.left = `${pct * 100}%`;
    hover.textContent = `${time} / Frame ${frame + 1}`;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function showToast(title, message, type = 'info', duration = 5000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<div class="toast-title">${title}</div><div class="toast-message">${message}</div>`;
    // Click to dismiss
    toast.style.cursor = 'pointer';
    toast.addEventListener('click', () => {
        toast.style.animation = 'toast-out 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    });
    container.appendChild(toast);
    if (duration > 0) {
        setTimeout(() => {
            toast.style.animation = 'toast-out 0.3s ease-out forwards';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

// Timeline interaction
document.addEventListener('DOMContentLoaded', () => {
    // --- Resizable sidebars ---
    document.querySelectorAll('.sidebar-resize-handle').forEach((handle) => {
        const sidebar = handle.previousElementSibling;
        if (!sidebar || !sidebar.classList.contains('sidebar')) return;
        let startX, startW;
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startX = e.clientX;
            startW = sidebar.offsetWidth;
            handle.classList.add('dragging');
            function onMove(e) {
                const w = Math.max(240, Math.min(startW + e.clientX - startX, window.innerWidth * 0.5));
                sidebar.style.width = w + 'px';
            }
            function onUp() {
                handle.classList.remove('dragging');
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    });

    const timelineContainer = document.getElementById('timeline-container');
    const timeline = document.getElementById('timeline');
    const scrubber = document.getElementById('timeline-scrubber');

    // Click to seek (but not if we were dragging trim handles)
    timeline.addEventListener('click', (e) => {
        if (!isDragging && !justFinishedTrimDrag) {
            // Check if click was on a trim handle
            if (e.target.classList.contains('trim-handle')) return;
            seekTimeline(getFrameFromTimelineEvent(e));
        }
    });

    // Hover preview
    timelineContainer.addEventListener('mousemove', updateHoverPreview);

    // Drag scrubber
    scrubber.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDragging = true;
        document.body.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging && currentDataset && currentEpisode !== null) {
            seekTimeline(getFrameFromTimelineEvent(e));
        }
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            document.body.style.cursor = '';
        }
        if (isDraggingTrimLeft || isDraggingTrimRight) {
            isDraggingTrimLeft = false;
            isDraggingTrimRight = false;
            justFinishedTrimDrag = true;
            document.body.style.cursor = '';
            // Auto-save the trim
            saveTrim();
            // Reset after a short delay to allow click event to check
            setTimeout(() => { justFinishedTrimDrag = false; }, 50);
        }
    });

    // Trim handle drag
    const trimHandleLeft = document.getElementById('trim-handle-left');
    const trimHandleRight = document.getElementById('trim-handle-right');

    trimHandleLeft.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDraggingTrimLeft = true;
        document.body.style.cursor = 'ew-resize';
    });

    trimHandleRight.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDraggingTrimRight = true;
        document.body.style.cursor = 'ew-resize';
    });

    document.addEventListener('mousemove', (e) => {
        if (!currentDataset || currentEpisode === null) return;

        if (isDraggingTrimLeft) {
            const frame = getFrameFromTimelineEvent(e);
            trimStart = Math.max(0, Math.min(frame, trimEnd - 1));
            updateTrimDisplay();
        } else if (isDraggingTrimRight) {
            const frame = getFrameFromTimelineEvent(e);
            // trimEnd is exclusive, so we add 1 to the clicked frame
            trimEnd = Math.max(trimStart + 1, Math.min(frame + 1, totalFrames));
            updateTrimDisplay();
        }
    });
});

// Keyboard controls
// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
    // Re-scan sources when switching to data tab (picks up newly recorded datasets)
    if (tabName === 'data' && typeof window.refreshExpandedSources === 'function') {
        window.refreshExpandedSources();
    }
    // Notify robot tab
    if (tabName === 'robot' && typeof robotTabInit === 'function') {
        robotTabInit();
    }
    // Notify run tab
    if (tabName === 'run' && typeof runTabInit === 'function') {
        runTabInit();
    }
    // Notify model tab
    if (tabName === 'model' && typeof modelTabInit === 'function') {
        modelTabInit();
    }
    // Stop camera preview when leaving robot tab
    if (tabName !== 'robot' && typeof stopCameraPreview === 'function') {
        stopCameraPreview();
    }
    // Disconnect SSE when leaving run tab (but don't kill process)
    if (tabName !== 'run' && typeof disconnectOutputSSE === 'function') {
        disconnectOutputSSE();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    // Only handle data-tab shortcuts when data tab is active
    const activeTab = document.querySelector('.tab.active')?.dataset.tab;
    if (activeTab !== 'data') return;
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        loadAllFrames(currentFrame - (e.shiftKey ? 10 : 1));
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        loadAllFrames(currentFrame + (e.shiftKey ? 10 : 1));
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        navigateEpisode(-1);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        navigateEpisode(1);
    } else if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
    } else if (e.key === 'Home') {
        e.preventDefault();
        loadAllFrames(0);
    } else if (e.key === 'End') {
        e.preventDefault();
        loadAllFrames(totalFrames - 1);
    } else if (e.key === 'Delete' && currentDataset && currentEpisode !== null) {
        e.preventDefault();
        deleteCurrentEpisode();
    } else if (e.key === 'r' && currentDataset && currentEpisode !== null) {
        e.preventDefault();
        resetTrim();
    } else if (e.key === 'Escape') {
        hideContextMenu();
    }
});

function navigateEpisode(direction) {
    if (!currentDataset || currentEpisode === null) return;
    const dsEpisodes = episodes[currentDataset] || [];
    const newIndex = currentEpisode + direction;
    if (newIndex >= 0 && newIndex < dsEpisodes.length) {
        const ep = dsEpisodes.find(e => e.episode_index === newIndex);
        if (ep) selectEpisode(currentDataset, newIndex, ep.video_length || ep.length);
    }
}

// Context menu
// Position a context menu so it stays inside the viewport. The menu must
// already be `visible` (or have its dimensions otherwise readable) before
// calling — we measure with getBoundingClientRect after a forced layout.
function _positionContextMenu(menu, clientX, clientY) {
    const margin = 4;
    const rect = menu.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    let left = clientX;
    let top = clientY;
    if (left + rect.width + margin > vw) left = Math.max(margin, vw - rect.width - margin);
    if (top + rect.height + margin > vh) top = Math.max(margin, vh - rect.height - margin);
    menu.style.left = left + 'px';
    menu.style.top = top + 'px';
}

function showContextMenu(e, datasetId, episodeIndex) {
    e.preventDefault();
    e.stopPropagation();

    contextMenuTarget = { datasetId, episodeIndex };
    const menu = document.getElementById('context-menu');
    const isDeleted = isEpisodeDeleted(datasetId, episodeIndex);
    const isTrimmed = isEpisodeTrimmed(datasetId, episodeIndex);

    // Show/hide appropriate menu items
    menu.querySelectorAll('.context-menu-item').forEach(item => {
        const action = item.getAttribute('onclick').match(/contextAction\('(\w+)'\)/)?.[1];
        if (action === 'delete') item.style.display = isDeleted ? 'none' : 'block';
        if (action === 'undelete') item.style.display = isDeleted ? 'block' : 'none';
        if (action === 'cleartrim') item.style.display = isTrimmed ? 'block' : 'none';
    });

    // Make visible BEFORE measuring so getBoundingClientRect returns real dims.
    menu.classList.add('visible');
    _positionContextMenu(menu, e.clientX, e.clientY);
}

function hideContextMenu() {
    document.getElementById('context-menu').classList.remove('visible');
    document.getElementById('folder-context-menu').classList.remove('visible');
    contextMenuTarget = null;
    _folderContextPath = null;
}

document.addEventListener('click', hideContextMenu);

// Folder context menu (source folders + datasets + model runs)
let _folderContextPath = null;
let _folderContextIsModelRun = false;

function showFolderContextMenu(e, path, isModelRun) {
    e.preventDefault();
    e.stopPropagation();
    _folderContextPath = path;
    _folderContextIsModelRun = !!isModelRun;
    const menu = document.getElementById('folder-context-menu');
    // Show/hide model-run-specific items
    const testItem = document.getElementById('folder-ctx-test-on-robot');
    const testSep = document.getElementById('folder-ctx-test-separator');
    if (testItem) testItem.style.display = _folderContextIsModelRun ? '' : 'none';
    if (testSep) testSep.style.display = _folderContextIsModelRun ? '' : 'none';
    // Show/hide merge-into for opened datasets with 2+ datasets open
    const isOpenedDataset = !!datasets[path];
    const hasMultipleDatasets = Object.keys(datasets).length >= 2;
    const mergeItem = document.getElementById('folder-ctx-merge-into');
    const mergeSep = document.getElementById('folder-ctx-merge-separator');
    if (mergeItem) mergeItem.style.display = (isOpenedDataset && hasMultipleDatasets) ? '' : 'none';
    if (mergeSep) mergeSep.style.display = (isOpenedDataset && hasMultipleDatasets) ? '' : 'none';
    // Show/hide Hub upload/download for opened datasets
    const hubUpload = document.getElementById('folder-ctx-hub-upload');
    const hubDownload = document.getElementById('folder-ctx-hub-download');
    const hubSep = document.getElementById('folder-ctx-hub-separator');
    if (hubUpload) hubUpload.style.display = isOpenedDataset ? '' : 'none';
    if (hubDownload) hubDownload.style.display = isOpenedDataset ? '' : 'none';
    if (hubSep) hubSep.style.display = isOpenedDataset ? '' : 'none';
    menu.classList.add('visible');
    _positionContextMenu(menu, e.clientX, e.clientY);
}

function folderContextAction(action) {
    if (!_folderContextPath) return;
    if (action === 'open-in-files') {
        fetch('/api/datasets/open-in-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: _folderContextPath })
        }).catch(e => console.error('Failed to open file manager:', e));
    } else if (action === 'test-on-robot') {
        if (typeof testModelOnRobot === 'function') {
            testModelOnRobot(_folderContextPath);
        }
    } else if (action === 'merge-into') {
        openMergeModal(_folderContextPath);
    } else if (action === 'hub-upload') {
        hubUploadDataset(_folderContextPath);
    } else if (action === 'hub-download') {
        hubDownloadDataset(_folderContextPath);
    }
    hideContextMenu();
}

// --- Merge Into modal ---
let _mergeSourceId = null;

function openMergeModal(sourceDatasetId) {
    _mergeSourceId = sourceDatasetId;
    const sourceDs = datasets[sourceDatasetId];
    if (!sourceDs) return;

    document.getElementById('merge-source-name').textContent = sourceDs.repo_id;
    document.getElementById('merge-status').textContent = '';
    document.getElementById('merge-execute-btn').disabled = false;

    // Populate target dropdown with other opened datasets
    const select = document.getElementById('merge-target-select');
    select.innerHTML = '';
    for (const [id, ds] of Object.entries(datasets)) {
        if (id === sourceDatasetId) continue;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${ds.repo_id} (${ds.total_episodes} ep)`;
        select.appendChild(opt);
    }

    updateMergePreview();
    const overlay = document.getElementById('merge-modal-overlay');
    overlay.style.display = 'flex';
}

let _mergeForce = false;

function updateMergePreview() {
    const preview = document.getElementById('merge-preview');
    const diffPanel = document.getElementById('merge-diff-panel');
    const targetId = document.getElementById('merge-target-select').value;
    const sourceDs = _mergeSourceId ? datasets[_mergeSourceId] : null;
    const targetDs = targetId ? datasets[targetId] : null;
    if (!sourceDs || !targetDs) { preview.textContent = ''; diffPanel.style.display = 'none'; return; }

    const srcEps = sourceDs.total_episodes;
    const tgtEps = targetDs.total_episodes;
    preview.innerHTML =
        `<strong>${targetDs.repo_id}</strong> will go from ${tgtEps} to ${tgtEps + srcEps} episodes ` +
        `(+${srcEps} from ${sourceDs.repo_id}).<br>` +
        `This modifies the target dataset on disk.`;

    // Reset diff panel and force state on target change
    diffPanel.style.display = 'none';
    _mergeForce = false;
    const btn = document.getElementById('merge-execute-btn');
    btn.textContent = 'Merge (modifies target)';
    btn.style.background = '#c24038';
    document.getElementById('merge-status').textContent = '';

    // Run validation in background
    _validateMerge(sourceDs, targetDs, targetId);
}

async function _validateMerge(sourceDs, targetDs, targetId) {
    try {
        const res = await fetch('/api/edits/merge-into/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_dataset_id: _mergeSourceId,
                target_dataset_id: targetId,
            })
        });
        if (!res.ok) return;
        const data = await res.json();
        _renderMergeDiff(data);
    } catch (e) { /* ignore */ }
}

function _renderMergeDiff(validation) {
    const diffPanel = document.getElementById('merge-diff-panel');
    const btn = document.getElementById('merge-execute-btn');

    if (validation.compatible) {
        diffPanel.style.display = 'none';
        _mergeForce = false;
        btn.textContent = 'Merge (modifies target)';
        btn.style.background = '#c24038';
        btn.disabled = false;
        return;
    }

    // Build diff HTML
    let html = '';
    for (const m of validation.mismatches) {
        if (m.field === 'features') {
            if (m.target_only.length) {
                html += `<div class="merge-diff-section"><span class="merge-diff-label">Only in target:</span>`;
                for (const k of m.target_only)
                    html += `<span class="merge-diff-removed">${_esc(k)}</span>`;
                html += `</div>`;
            }
            if (m.source_only.length) {
                html += `<div class="merge-diff-section"><span class="merge-diff-label">Only in source:</span>`;
                for (const k of m.source_only)
                    html += `<span class="merge-diff-added">${_esc(k)}</span>`;
                html += `</div>`;
            }
            if (Object.keys(m.shared_diff).length) {
                html += `<div class="merge-diff-section"><span class="merge-diff-label">Different definitions:</span>`;
                for (const [k, v] of Object.entries(m.shared_diff)) {
                    html += `<details><summary>${_esc(k)}</summary>` +
                        `<div class="merge-diff-json">` +
                        `<div class="merge-diff-removed"><strong>target:</strong><pre>${_esc(JSON.stringify(v.target, null, 2))}</pre></div>` +
                        `<div class="merge-diff-added"><strong>source:</strong><pre>${_esc(JSON.stringify(v.source, null, 2))}</pre></div>` +
                        `</div></details>`;
                }
                html += `</div>`;
            }
        } else {
            html += `<div class="merge-diff-section">` +
                `<span class="merge-diff-label">${_esc(m.field)}:</span> ` +
                `<span class="merge-diff-removed">${_esc(String(m.target))}</span> (target) vs ` +
                `<span class="merge-diff-added">${_esc(String(m.source))}</span> (source)` +
                `</div>`;
        }
    }

    diffPanel.innerHTML = `<div class="merge-diff-header">Mismatches found</div>${html}`;
    diffPanel.style.display = 'block';

    // Switch button to force mode
    _mergeForce = true;
    btn.textContent = 'Force merge (skip validation)';
    btn.style.background = '#8b4513';
    btn.disabled = false;
}

function _esc(s) { const d = document.createElement('span'); d.textContent = s; return d.innerHTML; }

function closeMergeModal() {
    document.getElementById('merge-modal-overlay').style.display = 'none';
    document.getElementById('merge-diff-panel').style.display = 'none';
    _mergeSourceId = null;
    _mergeForce = false;
}

async function executeMerge() {
    const targetId = document.getElementById('merge-target-select').value;
    if (!_mergeSourceId || !targetId) return;

    const sourceDs = datasets[_mergeSourceId];
    const targetDs = datasets[targetId];

    const forceLabel = _mergeForce ? '\n\nWARNING: Skipping validation - features/metadata may differ!' : '';
    if (!confirm(
        `Merge ${sourceDs.total_episodes} episodes from "${sourceDs.repo_id}" ` +
        `into "${targetDs.repo_id}"?\n\n` +
        `This will modify "${targetDs.repo_id}" on disk.${forceLabel}`
    )) return;

    const btn = document.getElementById('merge-execute-btn');
    const status = document.getElementById('merge-status');
    btn.disabled = true;
    status.textContent = 'Merging...';

    try {
        const res = await fetch('/api/edits/merge-into', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_dataset_id: _mergeSourceId,
                target_dataset_id: targetId,
                force: _mergeForce,
            })
        });

        if (res.status === 423) {
            status.textContent = 'Dataset is busy, please wait.';
            btn.disabled = false;
            return;
        }

        const data = await res.json();
        if (!res.ok) {
            status.textContent = data.detail || 'Merge failed';
            btn.disabled = false;
            return;
        }

        closeMergeModal();
        showToast('Merge Complete', data.message, 'info');

        // Refresh the target dataset's episodes in the tree
        try {
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(targetId)}/episodes`);
            if (epRes.ok) {
                episodes[targetId] = await epRes.json();
                datasets[targetId].total_episodes = episodes[targetId].length;
                datasets[targetId].total_frames = episodes[targetId].reduce((s, e) => s + e.length, 0);
            }
        } catch (e) { /* ignore */ }
        renderTree();
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        btn.disabled = false;
    }
}

// Close merge modal on overlay click
document.addEventListener('click', (e) => {
    const overlay = document.getElementById('merge-modal-overlay');
    if (e.target === overlay) closeMergeModal();
});

function contextAction(action) {
    if (!contextMenuTarget) return;
    const { datasetId, episodeIndex } = contextMenuTarget;

    if (action === 'view') {
        const ep = episodes[datasetId]?.find(e => e.episode_index === episodeIndex);
        if (ep) selectEpisode(datasetId, episodeIndex, ep.video_length || ep.length);
    } else if (action === 'rerun') {
        const ep = episodes[datasetId]?.find(e => e.episode_index === episodeIndex);
        if (ep) {
            selectEpisode(datasetId, episodeIndex, ep.video_length || ep.length);
            launchRerun();
        }
    } else if (action === 'replay') {
        // Switch to Run tab → Replay workflow with this episode pre-selected
        if (typeof selectWorkflow === 'function') selectWorkflow('replay');
        switchTab('run');
        // After tab init renders the form, select the right episode
        setTimeout(() => {
            const sel = document.getElementById('run-replay-episode');
            if (sel) {
                const val = `${datasetId}:${episodeIndex}`;
                sel.value = val;
                if (typeof _onReplayEpisodeChange === 'function') _onReplayEpisodeChange();
            }
        }, 50);
    } else if (action === 'delete') {
        markEpisodeDeleted(datasetId, episodeIndex);
    } else if (action === 'undelete') {
        unmarkEpisodeDeleted(datasetId, episodeIndex);
    } else if (action === 'cleartrim') {
        clearEpisodeTrim(datasetId, episodeIndex);
    }

    hideContextMenu();
}

async function clearEpisodeTrim(datasetId, episodeIndex) {
    // Find and remove the trim edit
    const editIndex = pendingEdits.findIndex(
        e => e.dataset_id === datasetId && e.episode_index === episodeIndex && e.edit_type === 'trim'
    );
    if (editIndex >= 0) {
        try {
            const res = await fetch(`/api/edits/${editIndex}`, { method: 'DELETE' });
            if (!res.ok) throw new Error(await res.text());
            await refreshPendingEdits();
            setStatus(`Trim cleared for episode ${episodeIndex}`);
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }
}

// Edit operations
async function markEpisodeDeleted(datasetId, episodeIndex) {
    if (datasetBusy) { setStatus('Dataset is busy, please wait'); return; }
    try {
        const res = await fetch('/api/edits/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: datasetId, episode_index: episodeIndex })
        });
        if (res.status === 423) { setStatus('Dataset is busy, please wait'); return; }
        if (!res.ok) throw new Error(await res.text());
        await refreshPendingEdits();
        setStatus(`Episode ${episodeIndex} marked for deletion`);
    } catch (e) {
        setStatus('Error: ' + e.message);
    }
}

async function unmarkEpisodeDeleted(datasetId, episodeIndex) {
    if (datasetBusy) { setStatus('Dataset is busy, please wait'); return; }
    // Find and remove the delete edit
    const editIndex = pendingEdits.findIndex(
        e => e.dataset_id === datasetId && e.episode_index === episodeIndex && e.edit_type === 'delete'
    );
    if (editIndex >= 0) {
        try {
            const res = await fetch(`/api/edits/${editIndex}`, { method: 'DELETE' });
            if (res.status === 423) { setStatus('Dataset is busy, please wait'); return; }
            if (!res.ok) throw new Error(await res.text());
            await refreshPendingEdits();
            setStatus(`Episode ${episodeIndex} restored`);
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }
}

function deleteCurrentEpisode() {
    if (currentDataset && currentEpisode !== null) {
        if (isEpisodeDeleted(currentDataset, currentEpisode)) {
            unmarkEpisodeDeleted(currentDataset, currentEpisode);
        } else {
            markEpisodeDeleted(currentDataset, currentEpisode);
        }
    }
}

async function refreshPendingEdits() {
    try {
        const res = await fetch('/api/edits');
        const data = await res.json();
        pendingEdits = data.edits;
        renderTree();
        loadTrimForCurrentEpisode();
        if (window.FeatureEditing) window.FeatureEditing.onPendingEditsChanged();
    } catch (e) {
        console.error('Failed to refresh edits:', e);
    }
}

function updateEditsBar() {
    const bar = document.getElementById('edits-bar');
    const count = document.getElementById('edits-count');
    if (pendingEdits.length > 0) {
        bar.classList.add('visible');
        count.textContent = `${pendingEdits.length} pending edit${pendingEdits.length > 1 ? 's' : ''}`;
    } else {
        bar.classList.remove('visible');
    }
}

let datasetBusy = false;

function setEditingEnabled(enabled) {
    datasetBusy = !enabled;
    const bar = document.getElementById('edits-bar');
    if (bar) {
        bar.querySelectorAll('button').forEach(btn => btn.disabled = !enabled);
    }
}

async function discardEdits() {
    if (!confirm('Discard all pending edits?')) return;
    setEditingEnabled(false);
    try {
        const res = await fetch('/api/edits/discard', { method: 'POST' });
        if (res.status === 423) {
            setStatus('Dataset is busy, please wait');
            return;
        }
        if (!res.ok) throw new Error(await res.text());
        await refreshPendingEdits();
        setStatus('All edits discarded');
    } catch (e) {
        setStatus('Error: ' + e.message);
    } finally {
        setEditingEnabled(true);
    }
}

async function applyEdits() {
    if (!currentDataset) {
        setStatus('No dataset selected');
        return;
    }
    if (!confirm(
        `Apply ${pendingEdits.length} edit(s) to disk? This cannot be undone.\n\n` +
        `Pause any training jobs reading this dataset before continuing — ` +
        `the GUI server serializes its own writes, but external readers see ` +
        `torn state across shards mid-Save.`
    )) return;

    setEditingEnabled(false);
    setStatus('Applying edits...');
    try {
        const res = await fetch(`/api/edits/apply?dataset_id=${encodeURIComponent(currentDataset)}`, {
            method: 'POST'
        });
        if (res.status === 423) {
            setStatus('Dataset is busy, please wait');
            return;
        }
        const data = await res.json();
        if (data.status === 'ok' || data.status === 'partial') {
            // Reload dataset episodes
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(currentDataset)}/episodes`);
            episodes[currentDataset] = await epRes.json();
            // Sync episode count after edits may have changed it
            if (datasets[currentDataset]) {
                datasets[currentDataset].total_episodes = episodes[currentDataset].length;
            }
            await refreshPendingEdits();
            if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();

            // Re-select current episode (or nearest neighbour if deleted)
            if (currentDataset && currentEpisode !== null) {
                const epList = episodes[currentDataset] || [];
                const stillExists = epList.find(e => e.episode_index === currentEpisode);
                if (stillExists) {
                    // Re-select to refresh view (e.g. after trim changed length)
                    selectEpisode(currentDataset, currentEpisode, stillExists.video_length || stillExists.length);
                } else if (epList.length > 0) {
                    // Select nearest neighbour
                    const nearest = epList.reduce((best, e) =>
                        Math.abs(e.episode_index - currentEpisode) < Math.abs(best.episode_index - currentEpisode) ? e : best
                    );
                    selectEpisode(currentDataset, nearest.episode_index, nearest.video_length || nearest.length);
                } else {
                    currentEpisode = null;
                    renderCameraGrid();
                }
            }

            setStatus(data.message);
        } else {
            throw new Error(data.message);
        }
    } catch (e) {
        setStatus('Error: ' + e.message);
    } finally {
        setEditingEnabled(true);
    }
}

// Trim functions
function updateTrimDisplay() {
    const cutLeft = document.getElementById('trim-cut-left');
    const cutRight = document.getElementById('trim-cut-right');

    if (!currentDataset || currentEpisode === null || totalFrames === 0) {
        document.getElementById('trim-region').classList.remove('visible');
        document.getElementById('trim-controls').classList.remove('visible');
        cutLeft.classList.remove('visible');
        cutRight.classList.remove('visible');
        return;
    }

    const region = document.getElementById('trim-region');
    const leftPct = (trimStart / (totalFrames - 1)) * 100;
    const rightPct = ((trimEnd - 1) / (totalFrames - 1)) * 100;
    const widthPct = rightPct - leftPct;

    region.style.left = `${leftPct}%`;
    region.style.width = `${widthPct}%`;
    region.classList.add('visible');

    // Show cut zones (red tint for regions that will be removed)
    if (trimStart > 0) {
        cutLeft.style.width = `${leftPct}%`;
        cutLeft.classList.add('visible');
    } else {
        cutLeft.classList.remove('visible');
    }

    if (trimEnd < totalFrames) {
        cutRight.style.width = `${100 - rightPct}%`;
        cutRight.classList.add('visible');
    } else {
        cutRight.classList.remove('visible');
    }

    // Show trim controls if trim is different from full range
    const controls = document.getElementById('trim-controls');
    const info = document.getElementById('trim-info');
    if (trimStart > 0 || trimEnd < totalFrames) {
        const framesKept = trimEnd - trimStart;
        info.textContent = `Keep: frames ${trimStart}-${trimEnd - 1} (${framesKept} of ${totalFrames})`;
        controls.classList.add('visible');
    } else {
        controls.classList.remove('visible');
    }
}

function resetTrim() {
    trimStart = 0;
    trimEnd = totalFrames;
    updateTrimDisplay();
}

async function saveTrim() {
    if (!currentDataset || currentEpisode === null) return;
    if (datasetBusy) { setStatus('Dataset is busy, please wait'); return; }

    try {
        const res = await fetch('/api/edits/trim', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: currentDataset,
                episode_index: currentEpisode,
                start_frame: trimStart,
                end_frame: trimEnd
            })
        });
        if (res.status === 423) { setStatus('Dataset is busy, please wait'); return; }
        if (!res.ok) throw new Error(await res.text());
        await refreshPendingEdits();
        setStatus(`Trim set: keeping frames ${trimStart}-${trimEnd - 1} of episode ${currentEpisode}`);
    } catch (e) {
        setStatus('Error: ' + e.message);
    }
}

function loadTrimForCurrentEpisode() {
    if (!currentDataset || currentEpisode === null) return;

    // Check if there's an existing trim edit for this episode
    const existingTrim = pendingEdits.find(
        e => e.dataset_id === currentDataset && e.episode_index === currentEpisode && e.edit_type === 'trim'
    );

    if (existingTrim) {
        trimStart = existingTrim.params.start_frame;
        trimEnd = existingTrim.params.end_frame;
    } else {
        trimStart = 0;
        trimEnd = totalFrames;
    }
    updateTrimDisplay();
}

// Make state accessible for other scripts (run.js, etc.)
window.datasets = datasets;
window.episodes = episodes;
window.sourceDatasets = sourceDatasets;
window.refreshExpandedSources = async function() {
    for (const sourcePath of expandedSources) {
        scanSource(sourcePath);
    }
};
window.refreshOpenedDatasets = async function() {
    for (const id of Object.keys(datasets)) {
        try {
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(id)}/episodes`);
            if (epRes.ok) {
                episodes[id] = await epRes.json();
                datasets[id].total_episodes = episodes[id].length;
            }
        } catch (e) { /* ignore per-dataset errors */ }
    }
    renderTree();
    if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
};


// Restore previously opened datasets
async function restoreOpenedDatasets() {
    try {
        const res = await fetch('/api/datasets/previously-opened');
        if (!res.ok) return;
        const items = await res.json();
        for (const item of items) {
            try {
                await openDataset(item.root);
            } catch (e) {
                console.warn(`Failed to restore dataset ${item.root}:`, e);
            }
        }
    } catch (e) {
        console.warn('Failed to restore opened datasets:', e);
    }
}

// --- HuggingFace Hub operations ---

async function checkHubAuth() {
    try {
        const res = await fetch('/api/datasets/hub/auth-status');
        const data = await res.json();
        const el = document.getElementById('hf-auth-indicator');
        if (el) {
            el.textContent = data.logged_in ? `HF: @${data.username}` : 'HF: not logged in';
            el.style.color = data.logged_in ? 'var(--text-secondary, #888)' : 'var(--text-tertiary, #555)';
        }
    } catch (e) { /* ignore */ }
}

let _hubDatasetId = null;
let _hubAction = null;  // 'upload' | 'download' | 'open-sync'
let _hubOpenSyncCtx = null;  // { body, detail } for 'open-sync' mode
let _hubRepoInfoTimer = null;

function hubUploadDataset(datasetId) { openHubModal(datasetId, 'upload'); }
function hubDownloadDataset(datasetId) { openHubModal(datasetId, 'download'); }

// Enable/disable the Hub modal's primary button with a *visible* disabled
// state — the inline accent background overrides the browser's greyed-out
// styling, so a bare `disabled` looks identical to enabled. Dim + not-allowed
// cursor make a gated 'Download & Open' read as inert.
function setHubExecuteEnabled(enabled) {
    const b = document.getElementById('hub-execute-btn');
    b.disabled = !enabled;
    b.style.opacity = enabled ? '1' : '0.5';
    b.style.cursor = enabled ? 'pointer' : 'not-allowed';
}

function openHubModal(datasetId, action, ctx) {
    _hubDatasetId = datasetId;
    _hubAction = action;
    _hubOpenSyncCtx = action === 'open-sync' ? (ctx || null) : null;

    const ds = datasetId != null ? datasets[datasetId] : null;
    // Upload/download require an already-opened dataset; open-sync does not.
    if (action !== 'open-sync' && !ds) return;
    if (action === 'open-sync' && !_hubOpenSyncCtx) return;

    const titleEl = document.getElementById('hub-modal-title');
    const btn = document.getElementById('hub-execute-btn');
    const statusEl = document.getElementById('hub-status');
    const repoInput = document.getElementById('hub-repo-input');
    const localInfoEl = document.getElementById('hub-local-info');
    const repoInfoEl = document.getElementById('hub-repo-info');

    btn.disabled = false;
    statusEl.textContent = '';
    repoInfoEl.innerHTML = '<span style="color:var(--text-tertiary,#666)">Loading remote info...</span>';

    // Restore reusable modal chrome: the metadata-inconsistent open-sync state
    // hides the repo input, remote-info panel, and execute button; the
    // missing-files state dims the button until the repo is confirmed.
    document.getElementById('hub-repo-input-row').style.display = '';
    repoInfoEl.style.display = '';
    btn.style.display = '';
    setHubExecuteEnabled(true);

    if (action === 'upload') {
        titleEl.textContent = 'Upload to Hub';
        btn.textContent = 'Upload';
        btn.style.background = 'var(--accent, #0e639c)';
        repoInput.value = ds.repo_id;
        localInfoEl.innerHTML =
            `<strong>Local:</strong> ${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames<br>` +
            `<span style="color:var(--text-tertiary,#666)">${ds.root}</span>`;
    } else if (action === 'download') {
        titleEl.textContent = 'Download from Hub';
        btn.textContent = 'Download';
        btn.style.background = '#c24038';
        repoInput.value = ds.repo_id;
        localInfoEl.innerHTML =
            `<strong>Local:</strong> ${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames<br>` +
            `<span style="color:var(--text-tertiary,#666)">${ds.root}</span>`;
    } else if (action === 'open-sync') {
        const { detail } = _hubOpenSyncCtx;
        const probs = (detail.problems || []).slice(0, 5)
            .map(p => `<div style="color:var(--text-tertiary,#666); font-size:11px;">• ${p}</div>`).join('');
        const more = (detail.problems || []).length > 5
            ? `<div style="color:var(--text-tertiary,#666); font-size:11px;">• (and ${detail.problems.length - 5} more)</div>`
            : '';

        if (detail.kind === 'metadata_inconsistent') {
            // The metadata contradicts itself — not a download problem. State it
            // faithfully and drop the Hub chrome; only the Cancel button remains.
            titleEl.textContent = "Couldn't open dataset — metadata is inconsistent";
            document.getElementById('hub-repo-input-row').style.display = 'none';
            repoInfoEl.style.display = 'none';
            btn.style.display = 'none';
            repoInput.value = '';  // nothing to look up; keeps fetchHubRepoInfo a no-op
            localInfoEl.innerHTML =
                `<span style="color:var(--text-tertiary,#666)">${detail.local_path}</span>` +
                `<div style="margin-top:6px;"><strong>Problem:</strong></div>${probs}${more}` +
                `<div style="margin-top:8px; color:var(--text-tertiary,#666); font-size:11px;">` +
                `The dataset's <code>info.json</code> and its episode metadata table disagree, so it can't be ` +
                `opened. This isn't a missing-files problem — there's nothing to download.</div>`;
        } else {
            // Missing files — re-downloadable when a Hub copy exists. Gate the
            // button until fetchHubRepoInfo() confirms the repo exists.
            titleEl.textContent = 'Open dataset — local cache is incomplete';
            btn.textContent = 'Download & Open';
            btn.style.background = 'var(--accent, #0e639c)';
            setHubExecuteEnabled(false);  // gated until fetchHubRepoInfo confirms the repo
            repoInput.value = detail.repo_id || '';
            localInfoEl.innerHTML =
                `<strong>Local cache:</strong> incomplete<br>` +
                `<span style="color:var(--text-tertiary,#666)">${detail.local_path}</span>` +
                `<div style="margin-top:6px;"><strong>Missing:</strong></div>${probs}${more}` +
                `<div style="margin-top:8px; color:var(--text-tertiary,#666); font-size:11px;">` +
                `If this dataset is on the Hub, <em>Download &amp; Open</em> fetches the missing files into the path ` +
                `above, then opens it. Progress prints to the server terminal — this dialog stays open until done.</div>`;
        }
    }

    document.getElementById('hub-modal-overlay').style.display = 'flex';
    fetchHubRepoInfo();
    if (action !== 'open-sync') fetchHubDiff();
}

function closeHubModal() {
    document.getElementById('hub-modal-overlay').style.display = 'none';
    _hubDatasetId = null;
    _hubAction = null;
    _hubOpenSyncCtx = null;
}

function fetchHubRepoInfo() {
    clearTimeout(_hubRepoInfoTimer);
    _hubRepoInfoTimer = setTimeout(async () => {
        const repoId = document.getElementById('hub-repo-input').value.trim();
        const infoEl = document.getElementById('hub-repo-info');
        if (!repoId) { infoEl.innerHTML = ''; return; }

        infoEl.innerHTML = '<span style="color:var(--text-tertiary,#666)">Loading...</span>';
        try {
            const res = await fetch(`/api/datasets/hub/repo-info?repo_id=${encodeURIComponent(repoId)}`);
            const data = await res.json();
            const hubUrl = `https://huggingface.co/datasets/${repoId}`;
            const linkHtml = `<a href="${hubUrl}" target="_blank" rel="noopener noreferrer" style="color:#61afef; text-decoration:none;" title="Open on HuggingFace Hub">${repoId} ↗</a>`;
            if (!data.exists) {
                infoEl.innerHTML = _hubAction === 'upload'
                    ? `<span style="color:#e5c07b">New repo — will be created on upload</span><br><span style="color:var(--text-tertiary,#666)">URL: ${linkHtml}</span>`
                    : '<span style="color:#e06c75">Repo not found on Hub</span>';
                if (_hubAction === 'open-sync') {
                    // Not on the Hub → nothing to download. Keep the action
                    // blocked instead of offering a misleading 'Download & Open'.
                    setHubExecuteEnabled(false);
                    document.getElementById('hub-execute-btn').title =
                        "This repo isn't on the Hub — nothing to download.";
                    document.getElementById('hub-status').innerHTML =
                        '<span style="color:#e06c75">Not on the Hub — nothing to download.</span>';
                }
                return;
            }
            // Repo exists: for open-sync this makes 'Download & Open' a valid
            // action, so lift the gate set when the modal opened.
            if (_hubAction === 'open-sync') {
                setHubExecuteEnabled(true);
                document.getElementById('hub-execute-btn').title = '';
                document.getElementById('hub-status').textContent = '';
            }
            const epInfo = data.total_episodes != null
                ? `${data.total_episodes} episodes, ${data.total_frames?.toLocaleString() || '?'} frames`
                : `${data.files} files`;
            infoEl.innerHTML =
                `<strong>Remote:</strong> ${linkHtml}<br>` +
                `${epInfo}, ${data.total_size_mb} MB` +
                `${data.private ? ' (private)' : ''}<br>` +
                `Last modified: ${data.last_modified || 'unknown'}<br>` +
                `Downloads: ${data.downloads} | SHA: ${data.sha || '?'}`;
        } catch (e) {
            infoEl.innerHTML = `<span style="color:#e06c75">Failed to fetch info</span>`;
        }
        // Also refresh diff when repo changes
        fetchHubDiff();
    }, 400);
}

async function fetchHubDiff() {
    if (!_hubDatasetId) return;
    const repoId = document.getElementById('hub-repo-input').value.trim();
    const statusEl = document.getElementById('hub-status');
    if (!repoId) { statusEl.textContent = ''; return; }

    statusEl.innerHTML = '<span style="color:var(--text-tertiary,#666)">Comparing...</span>';
    try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(_hubDatasetId)}/hub/diff?repo_id=${encodeURIComponent(repoId)}`);
        const data = await res.json();
        if (data.status === 'error') {
            statusEl.textContent = data.message;
            return;
        }
        if (data.in_sync) {
            statusEl.innerHTML = '<span style="color:#98c379">In sync — no differences</span>';
            return;
        }
        let parts = [];
        if (data.modified.length > 0) parts.push(`${data.modified.length} modified`);
        if (data.local_only.length > 0) parts.push(`${data.local_only.length} local only`);
        if (data.remote_only.length > 0) parts.push(`${data.remote_only.length} remote only`);
        statusEl.innerHTML = `<span style="color:#e5c07b">${parts.join(', ')} (${data.unchanged} unchanged)</span>`;
    } catch (e) {
        statusEl.textContent = '';
    }
}

async function executeHubAction() {
    // open-sync uses _hubOpenSyncCtx instead of _hubDatasetId
    if (!_hubAction) return;
    if (_hubAction !== 'open-sync' && !_hubDatasetId) return;
    if (_hubAction === 'open-sync' && !_hubOpenSyncCtx) return;

    const repoId = document.getElementById('hub-repo-input').value.trim();
    if (!repoId) return;

    const btn = document.getElementById('hub-execute-btn');
    const status = document.getElementById('hub-status');
    btn.disabled = true;

    // 'open-sync' is a separate, synchronous flow — different endpoint
    // (the dataset-open one, not /hub/{upload,download}) and the response
    // body IS the opened dataset. Keep it blocking; promoting it to the
    // background-job pattern would require touching the open codepath.
    if (_hubAction === 'open-sync') {
        status.textContent = 'Downloading & opening…';
        try {
            const res = await fetch('/api/datasets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ..._hubOpenSyncCtx.body,
                    repo_id: repoId,
                    confirm_hub_sync: true,
                }),
            });
            if (!res.ok) {
                let msg = 'Open failed';
                try {
                    const data = await res.json();
                    msg = (data && data.detail) || msg;
                    if (typeof msg === 'object') msg = msg.message || JSON.stringify(msg);
                } catch (_) {}
                status.textContent = msg;
                btn.disabled = false;
                return;
            }
            const data = await res.json();
            closeHubModal();
            await _completeOpen(data);
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
            btn.disabled = false;
        }
        return;
    }

    // Upload / download: kick off a background job, close the modal
    // immediately, surface progress in the top-bar Transfers tray.
    const endpoint = `/api/datasets/${encodeURIComponent(_hubDatasetId)}/hub/${_hubAction}`;
    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId }),
        });

        if (res.status === 401) {
            status.textContent = 'Not logged in. Run `huggingface-cli login` in terminal.';
            btn.disabled = false;
            return;
        }
        if (res.status === 423) {
            status.textContent = 'Dataset is busy, please wait.';
            btn.disabled = false;
            return;
        }

        const data = await res.json();
        if (!res.ok) {
            // 409 with job_id = a Hub transfer is already running for this dataset.
            if (res.status === 409 && data?.detail?.job_id) {
                closeHubModal();
                Transfers.refreshNow();
                Transfers.openPopover();
                showToast('Transfer already running', 'See the Transfers tray for progress.', 'info', 4000);
                return;
            }
            // 409 with code=incomplete_local_state = the upload-time completeness
            // check found files present on the remote but missing locally. This is
            // the download-fail-then-upload guardrail. Ask the user before proceeding.
            if (res.status === 409 && data?.detail?.code === 'incomplete_local_state') {
                const missing = (data.detail.missing_locally || []).slice(0, 5);
                const incomplete = (data.detail.incomplete_locally || []).slice(0, 5);
                const detailLines = [];
                if (missing.length) detailLines.push('Missing: ' + missing.join(', '));
                if (incomplete.length) detailLines.push('Incomplete: ' + incomplete.join(', '));
                const ok = confirm(
                    'Your local copy is missing files that exist on the remote ' +
                    '(likely from an interrupted download). Uploading would push a ' +
                    'worse-than-remote state, but HF history preserves the old commit ' +
                    'so the prior state remains recoverable.\n\n' +
                    detailLines.join('\n') +
                    '\n\nUpload anyway?'
                );
                if (!ok) {
                    status.textContent = 'Cancelled. Re-download first to restore the missing files.';
                    btn.disabled = false;
                    return;
                }
                // Re-issue with confirm_force=true to bypass the guardrail.
                const force = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ repo_id: repoId, confirm_force: true }),
                });
                if (!force.ok) {
                    const fd = await force.json().catch(() => ({}));
                    status.textContent = fd?.detail?.message || fd?.detail || 'Upload failed';
                    btn.disabled = false;
                    return;
                }
                closeHubModal();
                Transfers.refreshNow();
                showToast('Upload started', 'Progress in the Transfers tray (top right).', 'info', 4000);
                return;
            }
            status.textContent = (data.detail && data.detail.message) || data.detail || 'Operation failed';
            btn.disabled = false;
            return;
        }

        // Job kicked off. Close modal, ping the tray, point the user at it.
        closeHubModal();
        Transfers.refreshNow();
        const verb = _hubAction === 'upload' ? 'Upload' : 'Download';
        showToast(`${verb} started`, 'Progress in the Transfers tray (top right).', 'info', 4000);
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        btn.disabled = false;
    }
}

// Hub modal: dismissal is intentionally Cancel-button-only. The transfer
// settings (repo id, sync direction) are easy to misclick away from when
// click-on-overlay closes the dialog, so we don't bind a backdrop handler.

// ── Hub Transfers tray ─────────────────────────────────────────────────
//
// Top-bar indicator + popover that lists every active or recently-finished
// Hub transfer. The single polling loop covers all transfers for the whole
// app — no more per-modal polling. Poll cadence:
//   - while any job is active: 1 Hz (counters tick once per file, faster
//     polling would just re-render the same snapshot)
//   - while idle (only-finished or empty): off — refreshNow() restarts it
//     when a fresh transfer is kicked off via executeHubAction
//
// Keyed by Transfers.* as a tiny module so executeHubAction + listeners
// don't have to know the internal state names. Globally exposed so the
// inline `onclick` handlers in the tab bar can call into it.

const Transfers = (function () {
    let _pollTimer = null;
    let _jobs = [];               // latest snapshot from /hub/jobs
    let _completionShown = new Set();  // job_ids we've already toasted
    let _popoverOpen = false;

    function _fmtBytes(n) {
        if (!n) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        let i = 0;
        while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
        return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
    }

    function _isActive(j) { return j.status === 'pending' || j.status === 'running'; }

    function _renderIndicator() {
        const ind = document.getElementById('transfers-indicator');
        const label = document.getElementById('transfers-indicator-label');
        if (!ind || !label) return;
        const active = _jobs.filter(_isActive);
        // Indicator is always present (the HTML omits `hidden`); we only
        // toggle the active styling + count badge based on job state. The
        // user can always find the popover entry point.
        ind.hidden = false;
        if (active.length > 0) {
            ind.classList.add('active');
            label.textContent = `Transfers · ${active.length}`;
        } else {
            ind.classList.remove('active');
            label.textContent = `Transfers`;
        }
    }

    function _renderPopover() {
        const list = document.getElementById('transfers-list');
        if (!list) return;
        if (_jobs.length === 0) {
            list.innerHTML =
                '<div class="transfers-empty" style="padding:14px 16px; color:var(--text-tertiary,#888); font-size:12px;">' +
                'No Hub transfers. Start one from a dataset\'s right-click menu &rarr; Upload / Download.' +
                '</div>';
            const clearBtn = document.querySelector('.transfers-clear-btn');
            if (clearBtn) clearBtn.disabled = true;
            return;
        }
        list.innerHTML = _jobs.map(_cardHtml).join('');
        const clearBtn = document.querySelector('.transfers-clear-btn');
        if (clearBtn) {
            const hasFinished = _jobs.some(j => !_isActive(j));
            clearBtn.disabled = !hasFinished;
        }
    }

    function _errorClassMessage(j) {
        // Prefer HF's own error text when present — for the fail-fast
        // cases (429, 401, 403) it's far more actionable than a canned
        // hint (e.g. carries the exact retry-after seconds and the
        // documented rate-limit caps). Fall back to a generic remediation
        // string when the worker didn't capture a specific message.
        if (!j.error && j.error_class !== 'cancelled') return '';
        switch (j.error_class) {
            case 'auth':
                return j.error || 'Authentication failed. Your HF token may be expired or lacks write permission. Run `huggingface-cli login` and click Retry.';
            case 'rate_limit':
                return j.error || 'Rate-limited by the Hub. Wait a few minutes and click Retry.';
            case 'bad_request':
                // Common cause we've observed: local HF upload cache
                // claims blobs are uploaded but HF doesn't have them
                // (state from a prior aborted upload). Verbatim message
                // first; if blank, point at the cache as the likely fix.
                return j.error || 'HF rejected the request. If this is a re-upload of a dataset that previously failed mid-flight, try clearing the dataset\'s `.cache/huggingface/upload/` directory before retrying.';
            case 'network':
                return `Network error: ${j.error}. Click Retry to resume.`;
            case 'cancelled':
                return 'Cancelled by user.';
            default:
                return j.error;
        }
    }

    function _cardHtml(j) {
        const dir = j.direction === 'upload' ? '▲ Upload' : '▼ Download';
        // Link to the PR for uploads (when one exists) so the user can
        // inspect the staged state. Falls back to the repo URL otherwise.
        const linkUrl = j.pr_url
            ? j.pr_url
            : `https://huggingface.co/datasets/${j.repo_id}`;
        const filesDone = j.files_done_estimate ?? 0;
        const filesTotal = j.files_total ?? 0;
        const bytesDone = j.bytes_done_estimate ?? 0;
        const bytesTotal = j.bytes_total ?? 0;
        const pct = filesTotal > 0
            ? Math.min(100, Math.round(100 * filesDone / filesTotal))
            : 0;

        // Action buttons depend on terminal-vs-active state. Cancel and
        // Discard get text labels because they affect remote state (kill
        // worker / close draft PR); Hide is icon-only because it's pure UI
        // dismissal with no consequences.
        let actions = '';
        let extra = '';
        if (_isActive(j)) {
            // Active: Cancel only (kills the worker subprocess).
            actions = `<button class="transfer-action-btn danger" type="button"
                onclick="Transfers.cancel('${j.job_id}')">Cancel</button>`;
            const stageLine = j.milestone
                ? `<div class="transfer-milestone">${j.milestone}</div>`
                : '';
            const curFile = j.current_file
                ? `<div class="transfer-current-file" title="${j.current_file}">${j.current_file}</div>`
                : '';
            extra = stageLine + curFile;
        } else if (j.status === 'complete') {
            // Complete: Hide is UI-only, nothing to clean up server-side.
            actions = `<button class="transfer-action-btn hide-btn" type="button"
                onclick="Transfers.hide('${j.job_id}')" title="Hide">✕</button>`;
            const bytesText = bytesDone > 0 ? ` · ${_fmtBytes(bytesDone)}` : '';
            extra = `<div class="transfer-msg complete">Done${bytesText}</div>`;
        } else {
            // Failed or cancelled: Retry (re-POST) + Discard (closes draft PR).
            actions =
                `<button class="transfer-action-btn" type="button"
                    onclick="Transfers.retry('${j.job_id}')">Retry</button>` +
                `<button class="transfer-action-btn danger" type="button"
                    onclick="Transfers.discard('${j.job_id}')">Discard</button>`;
            const msgClass = j.status === 'failed' ? 'failed' : 'cancelled';
            extra = `<div class="transfer-msg ${msgClass}">${_errorClassMessage(j) || 'Cancelled'}</div>`;
        }

        const showBar = filesTotal > 0 || _isActive(j);
        const progressLine = showBar
            ? `<div class="transfer-stats">${filesDone} / ${filesTotal} files` +
              (bytesTotal > 0 ? ` — ${_fmtBytes(bytesDone)} / ${_fmtBytes(bytesTotal)}` : '') +
              `<span class="pct">${pct}%</span></div>` +
              `<progress value="${filesDone}" max="${Math.max(1, filesTotal)}"></progress>`
            : '';

        // Short job-id prefix so a user clicking "Open log folder" can
        // identify which <job_id>.log file is theirs in the directory
        // listing. Full id is in the title attribute for copy/paste.
        const jobIdShort = (j.job_id || '').slice(0, 8);
        const jobIdChip = jobIdShort
            ? `<span class="transfer-jobid" title="job_id=${j.job_id}\nLog file: ${jobIdShort}…log" style="margin-left:auto; font-family:monospace; font-size:10px; color:var(--text-tertiary,#888);">${jobIdShort}</span>`
            : '';

        return (
            `<div class="transfer-card ${j.status}">` +
              `<div class="transfer-card-head">` +
                `<span class="transfer-direction">${dir}</span>` +
                `<a class="transfer-repo" href="${linkUrl}" target="_blank" rel="noopener noreferrer" title="${j.repo_id}">${j.repo_id}</a>` +
                jobIdChip +
                `<span class="transfer-actions">${actions}</span>` +
              `</div>` +
              progressLine +
              extra +
            `</div>`
        );
    }

    function _onJobsUpdated(prevJobs, jobs) {
        // Surface a one-shot toast for each transition into a terminal
        // state — but only when the popover is closed. When the popover is
        // open the card itself shows the new status, and an overlapping
        // toast on the same screen edge is redundant + visually noisy
        // (they share the top-right corner). We still record the job_id
        // in _completionShown so a later open/close doesn't re-toast.
        for (const j of jobs) {
            if (_isActive(j) || _completionShown.has(j.job_id)) continue;
            _completionShown.add(j.job_id);
            // Download still needs the post-completion refresh regardless
            // of toast visibility — it's not a notification, it's state sync.
            if (j.status === 'complete' && j.direction === 'download' && datasets[j.dataset_id]) {
                _refreshAfterDownload(j.dataset_id);
            }
            if (_popoverOpen) continue;
            const verb = j.direction === 'upload' ? 'Upload' : 'Download';
            const filesDone = j.files_done_estimate ?? 0;
            const bytesDone = j.bytes_done_estimate ?? 0;
            if (j.status === 'complete') {
                const bytesText = bytesDone > 0 ? `, ${_fmtBytes(bytesDone)}` : '';
                showToast(`${verb} complete`, `${j.repo_id} — ${filesDone} files${bytesText}`, 'info');
            } else if (j.status === 'failed') {
                showToast(`${verb} failed`, `${j.repo_id}: ${_errorClassMessage(j)}`, 'error', 8000);
            } else if (j.status === 'cancelled') {
                showToast(`${verb} cancelled`, j.repo_id, 'warning');
            }
        }
    }

    async function _refreshAfterDownload(datasetId) {
        try {
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/episodes`);
            if (epRes.ok) {
                episodes[datasetId] = await epRes.json();
                datasets[datasetId].total_episodes = episodes[datasetId].length;
                datasets[datasetId].total_frames = episodes[datasetId].reduce((s, e) => s + e.length, 0);
                renderTree();
            }
        } catch (e) { /* non-critical refresh */ }
    }

    async function poll() {
        try {
            const res = await fetch('/api/datasets/hub/jobs');
            if (!res.ok) return;
            const data = await res.json();
            const prev = _jobs;
            _jobs = data.jobs || [];
            _onJobsUpdated(prev, _jobs);
            _renderIndicator();
            if (_popoverOpen) _renderPopover();
        } catch (e) {
            // Network blip — keep last snapshot, try again next tick.
        }
        // Schedule next poll only if there's work to watch. When all jobs
        // are terminal, the indicator stays visible until dismissed but
        // we stop hammering the server.
        if (_pollTimer) clearTimeout(_pollTimer);
        _pollTimer = null;
        if (_jobs.some(_isActive)) {
            _pollTimer = setTimeout(poll, 1000);
        }
    }

    function refreshNow() {
        // Called when a new transfer is kicked off — restarts the poll
        // loop unconditionally (so a job initiated when the tray was
        // idle gets immediate attention).
        if (_pollTimer) clearTimeout(_pollTimer);
        _pollTimer = null;
        poll();
    }

    function openPopover() {
        _popoverOpen = true;
        const pop = document.getElementById('transfers-popover');
        if (pop) pop.hidden = false;
        _renderPopover();
    }

    function closePopover() {
        _popoverOpen = false;
        const pop = document.getElementById('transfers-popover');
        if (pop) pop.hidden = true;
    }

    function toggle() {
        if (_popoverOpen) closePopover();
        else openPopover();
    }

    async function cancel(jobId) {
        // Confirmation only if the transfer has actually started moving
        // bytes (active mid-flight). For a "still starting" / "just queued"
        // job the cancel is free of regret.
        const j = _jobs.find(x => x.job_id === jobId);
        if (j && (j.files_done_estimate ?? 0) > 0 && j.direction === 'upload') {
            const ok = confirm(
                'Cancel upload? Already-uploaded chunks stay on the server. ' +
                'The pending HF PR remains in draft so Retry can resume.'
            );
            if (!ok) return;
        }
        try {
            await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
            refreshNow();
        } catch (e) { /* ignored */ }
    }

    async function retry(jobId) {
        const j = _jobs.find(x => x.job_id === jobId);
        if (!j) return;
        // Retry is just re-POSTing the upload/download endpoint with the
        // same dataset+repo. The server detects the existing draft PR and
        // resumes into it via the reuse_pr_num path (transferring pr_num
        // ownership off the old terminal entry as a side effect, so the
        // follow-up dismiss below does NOT close the resumed PR).
        const endpoint = `/api/datasets/${encodeURIComponent(j.dataset_id)}/hub/${j.direction}`;
        const post = (body) => fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        try {
            let res = await post({ repo_id: j.repo_id });
            if (res.status === 409) {
                const data = await res.json().catch(() => ({}));
                const detail = data && data.detail;
                // Two distinct 409 shapes: "active job exists" carries job_id;
                // the completeness guardrail carries code: 'incomplete_local_state'.
                if (detail && detail.code === 'incomplete_local_state') {
                    const missing = detail.missing_locally || [];
                    const incomplete = detail.incomplete_locally || [];
                    const lines = [];
                    if (missing.length) lines.push('Missing: ' + missing.join(', '));
                    if (incomplete.length) lines.push('Incomplete: ' + incomplete.join(', '));
                    const ok = confirm(
                        'Local copy is missing files that exist on the remote.\n\n' +
                        lines.join('\n') +
                        '\n\nRetry the upload anyway?'
                    );
                    if (!ok) return;
                    res = await post({ repo_id: j.repo_id, confirm_force: true });
                    if (!res.ok) {
                        const fd = await res.json().catch(() => ({}));
                        showToast('Retry failed', fd?.detail?.message || fd?.detail || 'Could not restart transfer', 'error');
                        return;
                    }
                } else {
                    // Already running (concurrent retry); attach to the existing job.
                    refreshNow();
                    return;
                }
            } else if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                showToast('Retry failed', data.detail?.message || data.detail || 'Could not restart transfer', 'error');
                return;
            }
            // Drop the old terminal entry so the tray shows only the new attempt.
            // Safe: server already transferred pr_num ownership; this dismiss
            // will not close the resumed PR.
            await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/dismiss`, { method: 'POST' });
            refreshNow();
        } catch (e) {
            showToast('Retry failed', e.message, 'error');
        }
    }

    async function discard(jobId) {
        const j = _jobs.find(x => x.job_id === jobId);
        const isUpload = j && j.direction === 'upload';
        const hasPR = j && j.pr_num != null;
        if (isUpload && hasPR) {
            const ok = confirm(
                'Discard upload? The pending HF PR will be closed and ' +
                'partially uploaded data will be cleaned up. Resume will ' +
                'no longer be possible. Use Retry to resume instead.'
            );
            if (!ok) return;
        }
        try {
            const res = await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/dismiss`, { method: 'POST' });
            if (res.ok) refreshNow();
        } catch (e) { /* ignored */ }
    }

    async function hide(jobId) {
        // "Hide" is just a UI-only dismiss on a complete job; the server's
        // dismiss endpoint does the right thing (no PR to clean up since
        // the upload already merged).
        try {
            const res = await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/dismiss`, { method: 'POST' });
            if (res.ok) refreshNow();
        } catch (e) { /* ignored */ }
    }

    async function dismissAllFinished() {
        // Hide-all for complete cards + discard-all for failed/cancelled.
        // Iterates serially; the count is bounded by what's visible.
        const targets = _jobs.filter(j => !_isActive(j));
        // Discarding a failed/cancelled upload with a draft PR closes that
        // PR on HF (server's dismiss endpoint behavior). The single-card
        // Discard button confirms because of that; "Clear" must do the same
        // for the bulk path or the user can lose multiple resumable PRs in
        // one click. Complete uploads' PRs are already merged, and successful
        // retries have transferred PR ownership (pr_num cleared on the
        // source), so neither contributes to the count.
        const closingPRs = targets.filter(
            j => j.direction === 'upload'
                && (j.status === 'failed' || j.status === 'cancelled')
                && j.pr_num != null
        );
        if (closingPRs.length > 0) {
            const ok = confirm(
                `Discard ${closingPRs.length} failed/cancelled upload(s)? ` +
                `Their draft PRs on HF will be closed and resume will no longer ` +
                `be possible. Use Retry on each card to resume instead.`
            );
            if (!ok) return;
        }
        for (const j of targets) {
            try {
                await fetch(`/api/datasets/hub/progress/${encodeURIComponent(j.job_id)}/dismiss`, { method: 'POST' });
            } catch (e) { /* ignored */ }
        }
        refreshNow();
    }

    return { poll, refreshNow, openPopover, closePopover, toggle, cancel, retry, discard, hide, dismissAllFinished };
})();

// Global handles for the inline onclick attributes in index.html.
window.toggleTransfersPopover = () => Transfers.toggle();
window.dismissAllFinishedTransfers = () => Transfers.dismissAllFinished();
window.Transfers = Transfers;

// Opens the per-job log directory on the GUI host machine. Same constraint
// as the dataset "open in files" buttons: this is the host's filesystem,
// not the frontend's — fine when the GUI is running locally, degrades to
// a clear error toast when xdg-open isn't available (e.g. headless server).
window.openHubJobFolder = async () => {
    try {
        const res = await fetch('/api/datasets/hub/open-job-folder', { method: 'POST' });
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            showToast('Couldn\'t open folder', data?.detail || 'xdg-open failed on the GUI host', 'error', 6000);
        }
    } catch (e) {
        showToast('Couldn\'t open folder', e.message, 'error', 6000);
    }
};

// Click-outside closes the popover. Anchored to the indicator: if the
// click is on the indicator or inside the popover, leave it alone.
document.addEventListener('click', (e) => {
    const pop = document.getElementById('transfers-popover');
    const ind = document.getElementById('transfers-indicator');
    if (!pop || pop.hidden) return;
    if (ind && ind.contains(e.target)) return;
    if (pop.contains(e.target)) return;
    Transfers.closePopover();
});

// Initialize
refreshPendingEdits();
loadSources();
restoreOpenedDatasets();
checkHubAuth();
// Pick up any in-flight transfers from a prior session (the server keeps
// them until they finish + 30 min). One probe at startup is enough — if
// it returns active jobs the poll loop schedules itself thereafter.
Transfers.refreshNow();
