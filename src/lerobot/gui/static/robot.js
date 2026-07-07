/* LeRobot GUI - Robot Tab Logic */

let robotSchemas = null;
let teleopSchemas = null;
let robotProfiles = [];
let teleopProfiles = [];
let currentProfile = null; // {kind: 'robot'|'teleop', name, data}
let savedProfileData = null; // deep copy of data at last save/load — for dirty detection
let detectedCameras = [];
let previewInterval = null;
let scannedPorts = null; // {ports, allAssignments} from last scan
let robotTabInitialized = false;

// ============================================================================
// Initialization
// ============================================================================

async function robotTabInit() {
    if (robotTabInitialized) return;
    robotTabInitialized = true;
    await Promise.all([
        loadRobotSchemas(),
        loadTeleopSchemas(),
        loadRobotProfiles(),
        loadTeleopProfiles(),
    ]);
}

async function loadRobotSchemas() {
    try {
        const res = await fetch('/api/robot/schemas');
        robotSchemas = await res.json();
    } catch (e) {
        console.error('Failed to load robot schemas:', e);
    }
}

async function loadTeleopSchemas() {
    try {
        const res = await fetch('/api/robot/teleop-schemas');
        teleopSchemas = await res.json();
    } catch (e) {
        console.error('Failed to load teleop schemas:', e);
    }
}

// ============================================================================
// Profile list loading and rendering
// ============================================================================

async function loadRobotProfiles() {
    try {
        const res = await fetch('/api/robot/profiles');
        robotProfiles = await res.json();
    } catch (e) {
        robotProfiles = [];
    }
    renderRobotProfileList();
    if (typeof refreshRunProfileSelects === 'function') refreshRunProfileSelects();
}

async function loadTeleopProfiles() {
    try {
        const res = await fetch('/api/robot/teleop-profiles');
        teleopProfiles = await res.json();
    } catch (e) {
        teleopProfiles = [];
    }
    renderTeleopProfileList();
    if (typeof refreshRunProfileSelects === 'function') refreshRunProfileSelects();
}

function renderRobotProfileList() {
    const list = document.getElementById('robot-profile-list');
    if (!list) return;
    if (robotProfiles.length === 0) {
        list.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No profiles</div>';
        return;
    }
    list.innerHTML = robotProfiles.map(p => {
        const isActive = currentProfile?.name === p.name && currentProfile?.kind === 'robot';
        return `<div class="profile-item ${isActive ? 'active' : ''}"
                     onclick="selectProfile('robot', '${esc(p.name)}')">
            <span>${esc(p.name)}<span class="profile-type">${esc(p.type)}</span></span>
            <span class="profile-delete" onclick="event.stopPropagation(); deleteProfile('robot', '${esc(p.name)}')">&times;</span>
        </div>`;
    }).join('');
}

function renderTeleopProfileList() {
    const list = document.getElementById('teleop-profile-list');
    if (!list) return;
    if (teleopProfiles.length === 0) {
        list.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No profiles</div>';
        return;
    }
    list.innerHTML = teleopProfiles.map(p => {
        const isActive = currentProfile?.name === p.name && currentProfile?.kind === 'teleop';
        return `<div class="profile-item ${isActive ? 'active' : ''}"
                     onclick="selectProfile('teleop', '${esc(p.name)}')">
            <span>${esc(p.name)}<span class="profile-type">${esc(p.type)}</span></span>
            <span class="profile-delete" onclick="event.stopPropagation(); deleteProfile('teleop', '${esc(p.name)}')">&times;</span>
        </div>`;
    }).join('');
}

// ============================================================================
// Profile selection and editor
// ============================================================================

async function selectProfile(kind, name) {
    const endpoint = kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(name)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        currentProfile = { kind, name, data };
        savedProfileData = JSON.parse(JSON.stringify(data));
        // Side-load safe-trajectory metadata so the renderer can show
        // frame_count / duration without an extra async round-trip.
        if (kind === 'robot') {
            currentProfile._trajectoryMeta = null;
            _refreshTrajectoryMeta();
        }
        renderEditor();
        renderRobotProfileList();
        renderTeleopProfileList();
    } catch (e) {
        showToast('Error', `Failed to load profile: ${e.message}`, 'error');
    }
}

async function _refreshTrajectoryMeta() {
    if (!currentProfile || currentProfile.kind !== 'robot') return;
    const name = currentProfile.name;
    try {
        const res = await fetch(`/api/robot/trajectory-meta/${encodeURIComponent(name)}`);
        const meta = res.ok ? await res.json() : { exists: false };
        // Only apply if we're still on the same profile (user may have switched).
        if (currentProfile && currentProfile.name === name) {
            currentProfile._trajectoryMeta = meta;
            _rerender();
        }
    } catch (e) { /* leave meta null; UI shows empty state */ }
}

function renderEditor() {
    const emptyEl = document.getElementById('robot-empty');
    const editorEl = document.getElementById('robot-editor');
    if (!currentProfile) {
        emptyEl.style.display = 'flex';
        editorEl.style.display = 'none';
        return;
    }
    emptyEl.style.display = 'none';
    editorEl.style.display = 'block';

    const schemas = currentProfile.kind === 'robot' ? robotSchemas : teleopSchemas;
    const schema = schemas?.find(s => s.type_name === currentProfile.data.type);

    let html = '';

    // Header: name + actions
    html += '<div class="editor-header">';
    html += '<div class="editor-header-row">';
    html += `<h2>${esc(currentProfile.name)}</h2>`;
    html += `<button class="btn-small secondary" onclick="openProfileFolder()" title="Open in file manager">Open Folder</button>`;
    html += `<button class="btn-small secondary" onclick="renameProfile()" title="Rename profile">Rename</button>`;
    html += `<button class="btn-small danger" onclick="deleteProfile('${esc(currentProfile.kind)}', '${esc(currentProfile.name)}')" title="Delete profile">Delete</button>`;
    html += '</div>';
    html += '</div>';

    // Form fields (type selector as first row)
    html += '<div class="editor-form" id="editor-form">';
    html += `<label>Type</label>`;
    html += `<select onchange="changeProfileType(this.value)">`;
    if (schemas) {
        for (const s of schemas) {
            const sel = s.type_name === currentProfile.data.type ? 'selected' : '';
            html += `<option value="${esc(s.type_name)}" ${sel}>${esc(s.type_name)}</option>`;
        }
    }
    html += '</select>';
    if (schema) {
        for (const field of schema.fields) {
            const value = currentProfile.data.fields?.[field.name] ?? field.default ?? '';
            html += renderFormField(field, value);
        }
    }
    html += '</div>';

    // Cameras section (robot profiles only)
    if (currentProfile.kind === 'robot') {
        html += renderCamerasSection();
    }

    // Ports section (both robot and teleop — leaders have ports too)
    if (_getPortFields().length > 0) {
        html += renderPortsSection();
    }

    // Rest position section (robot profiles only)
    if (currentProfile.kind === 'robot') {
        html += renderRestPositionSection();
        html += renderSafeTrajectorySection();
    }

    // Save/discard bar (hidden until dirty)
    html += '<div class="editor-actions" id="editor-actions" style="display:none">';
    html += `<button class="btn-save" onclick="saveProfile()">Save Changes</button>`;
    html += `<button class="btn-discard" onclick="discardProfileChanges()">Discard</button>`;
    html += '</div>';

    editorEl.innerHTML = html;

    // Listen for input changes to track dirty state
    const form = document.getElementById('editor-form');
    if (form) {
        form.addEventListener('input', _updateDirtyState);
        form.addEventListener('change', _updateDirtyState);
    }
}

function renderFormField(field, value) {
    const id = `field-${field.name}`;
    // ``description`` is the dataclass field's metadata.description from
    // the backend schema. Render it as a ``title`` tooltip on both the
    // label and the form control so the user can hover either to read
    // it. Empty when the config doesn't supply metadata.
    const desc = typeof field.description === 'string' ? field.description : '';
    const labelTitle = desc ? ` title="${esc(desc)}"` : '';
    const controlTitle = desc ? ` title="${esc(desc)}"` : '';
    const label = `<label for="${id}"${labelTitle}>${esc(field.name)}${field.required ? ' *' : ''}</label>`;
    const typeStr = field.type_str.toLowerCase();

    // Literal["a", "b", ...] fields render as a dropdown of the allowed
    // values. Backend exposes `choices` whenever it detects a Literal
    // annotation, so any new enum-ish field auto-renders without a JS edit.
    // Per-choice tooltips come from ``field.choice_descriptions`` (also
    // schema-supplied via dataclass metadata) — useful for opaque
    // algorithm names where the choice value alone doesn't explain itself.
    if (Array.isArray(field.choices) && field.choices.length > 0) {
        const choiceDescs = (field.choice_descriptions && typeof field.choice_descriptions === 'object')
            ? field.choice_descriptions : {};
        const opts = field.choices
            .map(c => {
                const sel = String(value) === String(c) ? 'selected' : '';
                const cd = choiceDescs[c];
                const optTitle = cd ? ` title="${esc(cd)}"` : '';
                return `<option value="${esc(String(c))}" ${sel}${optTitle}>${esc(String(c))}</option>`;
            })
            .join('');
        return label + `<select id="${id}"${controlTitle}>${opts}</select>`;
    }

    if (typeStr === 'bool' || typeStr === 'bool | none') {
        const trueSelected = value === true ? 'selected' : '';
        const falseSelected = value === false ? 'selected' : '';
        const noneSelected = (value === null || value === '' || value === undefined) ? 'selected' : '';
        return label + `<select id="${id}"${controlTitle}>
            ${!field.required ? `<option value="null" ${noneSelected}>--</option>` : ''}
            <option value="true" ${trueSelected}>true</option>
            <option value="false" ${falseSelected}>false</option>
        </select>`;
    }

    if (typeStr.includes('int') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="1"${controlTitle}>`;
    }

    if (typeStr.includes('float') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="any"${controlTitle}>`;
    }

    // Default: text input
    const displayValue = (value === null || value === undefined) ? '' : String(value);
    return label + `<input type="text" id="${id}" value="${esc(displayValue)}"
        ${!field.required ? 'placeholder="(optional)"' : ''}${controlTitle}>`;
}

function changeProfileType(newType) {
    if (!currentProfile) return;
    currentProfile.data.type = newType;
    currentProfile.data.fields = {};
    currentProfile.data.cameras = {};
    currentProfile.data.rest_position = {};
    _rerender();
    _updateDirtyState();
}

// ============================================================================
// Dirty tracking
// ============================================================================

function _serializeProfile(data) {
    return JSON.stringify({
        type: data.type,
        name: data.name,
        fields: data.fields || {},
        cameras: data.cameras || {},
        rest_position: data.rest_position || {},
    });
}

function _collectFormFields() {
    const schemas = currentProfile.kind === 'robot' ? robotSchemas : teleopSchemas;
    const schema = schemas?.find(s => s.type_name === currentProfile.data.type);
    // Start from the loaded fields so any non-schema fields are preserved
    // — the backend's _SKIP_FIELDS (e.g. `calibration_dir`) hides certain
    // fields from the editing UI but they still need to round-trip through
    // save / launch. Starting from `{}` here would silently drop them and
    // any subsequent save would erase the JSON's calibration_dir.
    const fields = { ...(currentProfile?.data?.fields || {}) };
    if (schema) {
        for (const field of schema.fields) {
            const input = document.getElementById(`field-${field.name}`);
            if (!input) continue;
            fields[field.name] = parseFieldValue(field, input.value);
        }
    }
    return fields;
}

function _isDirty() {
    if (!currentProfile || !savedProfileData) return false;
    const current = _serializeProfile({
        ...currentProfile.data,
        name: currentProfile.name,
        fields: _collectFormFields(),
    });
    return current !== _serializeProfile(savedProfileData);
}

function _updateDirtyState() {
    const bar = document.getElementById('editor-actions');
    if (bar) bar.style.display = _isDirty() ? 'flex' : 'none';
}

function discardProfileChanges() {
    if (!currentProfile || !savedProfileData) return;
    currentProfile.data = JSON.parse(JSON.stringify(savedProfileData));
    _rerender();
}

// ============================================================================
// Profile CRUD
// ============================================================================

async function newRobotProfile() {
    const name = prompt('Robot profile name:');
    if (!name || !name.trim()) return;
    const defaultType = robotSchemas?.[0]?.type_name || 'so107_follower';
    const profile = { type: defaultType, name: name.trim(), fields: {}, cameras: {} };
    try {
        const res = await fetch('/api/robot/profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadRobotProfiles();
        await selectProfile('robot', name.trim());
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function newTeleopProfile() {
    const name = prompt('Teleop profile name:');
    if (!name || !name.trim()) return;
    const defaultType = teleopSchemas?.[0]?.type_name || 'keyboard';
    const profile = { type: defaultType, name: name.trim(), fields: {}, cameras: {} };
    try {
        const res = await fetch('/api/robot/teleop-profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadTeleopProfiles();
        await selectProfile('teleop', name.trim());
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function saveProfile() {
    if (!currentProfile) return;

    const payload = {
        ...currentProfile.data,
        name: currentProfile.name,
        fields: _collectFormFields(),
    };

    const endpoint = currentProfile.kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(currentProfile.name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        currentProfile.data = payload;
        savedProfileData = JSON.parse(JSON.stringify(payload));
        _updateDirtyState();
        showToast('Saved', `Profile "${currentProfile.name}" saved`, 'success');
        // Reload lists in case type changed
        if (currentProfile.kind === 'robot') await loadRobotProfiles();
        else await loadTeleopProfiles();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function openProfileFolder() {
    if (!currentProfile) return;
    try {
        const res = await fetch('/api/robot/open-in-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kind: currentProfile.kind })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Failed' }));
            showToast('Error', err.detail, 'error');
        }
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function renameProfile() {
    if (!currentProfile) return;
    const newName = prompt('New profile name:', currentProfile.name);
    if (!newName || !newName.trim() || newName.trim() === currentProfile.name) return;

    const endpoint = currentProfile.kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(currentProfile.name)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_name: newName.trim() }),
        });
        if (!res.ok) throw new Error(await res.text());
        const oldName = currentProfile.name;
        currentProfile.name = newName.trim();
        currentProfile.data.name = newName.trim();
        if (currentProfile.kind === 'robot') await loadRobotProfiles();
        else await loadTeleopProfiles();
        _rerender();
        showToast('Renamed', `"${oldName}" renamed to "${newName.trim()}"`, 'success');
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function deleteProfile(kind, name) {
    if (!confirm(`Delete profile "${name}"?`)) return;
    const endpoint = kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(await res.text());
        if (currentProfile?.name === name && currentProfile?.kind === kind) {
            currentProfile = null;
            renderEditor();
        }
        if (kind === 'robot') await loadRobotProfiles();
        else await loadTeleopProfiles();
        showToast('Deleted', `Profile "${name}" deleted`, 'info');
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

function parseFieldValue(field, rawValue) {
    if (rawValue === '' || rawValue === 'null') return null;
    const typeStr = field.type_str.toLowerCase();
    if (typeStr === 'bool' || typeStr === 'bool | none') {
        if (rawValue === 'true') return true;
        if (rawValue === 'false') return false;
        return null;
    }
    if (typeStr.includes('int') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        const n = parseInt(rawValue, 10);
        return isNaN(n) ? null : n;
    }
    if (typeStr.includes('float') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        const n = parseFloat(rawValue);
        return isNaN(n) ? null : n;
    }
    return rawValue;
}

// ============================================================================
// Camera detection and preview
// ============================================================================

function renderCamerasSection() {
    const cameras = currentProfile?.data?.cameras || {};
    const cameraEntries = Object.entries(cameras);

    let html = '<div class="camera-detect-section"><h3>Cameras</h3>';

    // Show assigned cameras with editable config
    if (cameraEntries.length > 0) {
        for (const [role, cam] of cameraEntries) {
            const camType = cam.type || 'unknown';
            const camId = cam.index_or_path ?? cam.serial_number_or_name ?? '?';
            const isRealsense = camType === 'intelrealsense';
            html += `<div class="camera-assigned">
                <div class="cam-header">
                    <span class="cam-role">${esc(role)}</span>
                    <span class="cam-info">${esc(camType)} - ${esc(String(camId))}</span>
                    <span class="cam-remove" onclick="removeCameraRole('${esc(role)}')" title="Remove">&times;</span>
                </div>
                <div class="cam-config">
                    <label>Width<input type="number" value="${cam.width || ''}" placeholder="auto" onchange="updateCameraConfig('${esc(role)}', 'width', this.value)"></label>
                    <label>Height<input type="number" value="${cam.height || ''}" placeholder="auto" onchange="updateCameraConfig('${esc(role)}', 'height', this.value)"></label>
                    <label>FPS<input type="number" value="${cam.fps || ''}" placeholder="auto" onchange="updateCameraConfig('${esc(role)}', 'fps', this.value)"></label>
                    ${!isRealsense ? `<label>Format<select onchange="updateCameraConfig('${esc(role)}', 'fourcc', this.value)">
                        <option value="" ${!cam.fourcc ? 'selected' : ''}>auto</option>
                        <option value="MJPG" ${cam.fourcc === 'MJPG' ? 'selected' : ''}>MJPG</option>
                        <option value="YUYV" ${cam.fourcc === 'YUYV' ? 'selected' : ''}>YUYV</option>
                        <option value="H264" ${cam.fourcc === 'H264' ? 'selected' : ''}>H264</option>
                    </select></label>` : ''}
                    ${isRealsense ? `<label>Depth<select onchange="updateCameraConfig('${esc(role)}', 'use_depth', this.value)">
                        <option value="false" ${!cam.use_depth ? 'selected' : ''}>off</option>
                        <option value="true" ${cam.use_depth ? 'selected' : ''}>on</option>
                    </select></label>` : ''}
                </div>
            </div>`;
        }
    }

    html += '<div class="camera-detect-actions">';
    html += '<button class="btn-small" id="detect-cameras-btn" onclick="detectCameras()">Detect Cameras</button>';
    html += `<button class="btn-small secondary" id="stop-cameras-btn" onclick="stopAllCameras()" style="display:${previewInterval ? 'inline-block' : 'none'}">Stop Preview</button>`;
    html += '</div>';
    html += '<div class="camera-preview-grid" id="camera-preview-grid"></div>';
    html += '</div>';
    return html;
}

async function detectCameras() {
    const btn = document.getElementById('detect-cameras-btn');
    if (btn) { btn.textContent = 'Detecting...'; btn.disabled = true; }

    try {
        const res = await fetch('/api/robot/detect-cameras', { method: 'POST' });
        detectedCameras = await res.json();
        renderCameraPreview();
        startCameraPreview();
    } catch (e) {
        showToast('Error', `Camera detection failed: ${e.message}`, 'error');
    } finally {
        if (btn) { btn.textContent = 'Detect Cameras'; btn.disabled = false; }
    }
}

function renderCameraPreview() {
    const grid = document.getElementById('camera-preview-grid');
    if (!grid) return;
    if (detectedCameras.length === 0) {
        grid.innerHTML = '<div style="color: #666; font-size: 13px; padding: 8px;">No cameras detected</div>';
        return;
    }

    // Default camera role names
    const roles = ['front', 'left_wrist', 'right_wrist', 'top'];

    // Build a reverse lookup: camera id -> assigned role
    const assignedRoles = {};
    const cameras = currentProfile?.data?.cameras || {};
    for (const [role, cam] of Object.entries(cameras)) {
        const camId = String(cam.index_or_path ?? cam.serial_number_or_name ?? '');
        if (camId) assignedRoles[camId] = role;
    }

    grid.innerHTML = detectedCameras.map((cam, i) => {
        const profile = cam.default_stream_profile || {};
        // Build identifier: serial number for RealSense, path for OpenCV
        const camId = String(cam.id || '');
        const identifier = cam.type === 'RealSense' ? `S/N: ${camId}` : camId;
        // Check if this camera is already assigned to a role
        const currentRole = assignedRoles[camId] || '';

        return `<div class="camera-preview-card">
            <img id="preview-cam-${i}" src="" alt="${esc(cam.name || 'Camera')}">
            <div class="camera-preview-info">
                <div>${esc(cam.name || `Camera ${i}`)}</div>
                <div class="cam-meta">${esc(cam.type)} | ${esc(identifier)} | ${profile.width || '?'}x${profile.height || '?'} @ ${Math.round(profile.fps) || '?'}fps</div>
                <select id="cam-role-${i}" onchange="assignCameraRole(${i}, this.value)">
                    <option value="">-- assign role --</option>
                    ${roles.map(r => `<option value="${r}" ${currentRole === r ? 'selected' : ''}>${r}</option>`).join('')}
                    <option value="__custom" ${currentRole && !roles.includes(currentRole) ? 'selected' : ''}>custom...</option>
                </select>
            </div>
        </div>`;
    }).join('');
}

function startCameraPreview() {
    stopCameraPreview();
    previewInterval = setInterval(() => {
        detectedCameras.forEach((_, i) => {
            const img = document.getElementById(`preview-cam-${i}`);
            if (img) {
                img.src = `/api/robot/camera-frame/${i}?t=${Date.now()}`;
            }
        });
    }, 100); // ~10fps
    const stopBtn = document.getElementById('stop-cameras-btn');
    if (stopBtn) stopBtn.style.display = 'inline-block';
}

function stopCameraPreview() {
    if (previewInterval) {
        clearInterval(previewInterval);
        previewInterval = null;
    }
    const stopBtn = document.getElementById('stop-cameras-btn');
    if (stopBtn) stopBtn.style.display = 'none';
}

async function stopAllCameras() {
    stopCameraPreview();
    try {
        await fetch('/api/robot/stop-cameras', { method: 'POST' });
    } catch (e) { /* ignore */ }
    detectedCameras = [];
    const grid = document.getElementById('camera-preview-grid');
    if (grid) grid.innerHTML = '';
}

function assignCameraRole(cameraIndex, role) {
    if (!currentProfile || !detectedCameras[cameraIndex]) return;

    if (role === '__custom') {
        role = prompt('Camera role name:');
        if (!role || !role.trim()) {
            // Reset dropdown
            const sel = document.getElementById(`cam-role-${cameraIndex}`);
            if (sel) sel.value = '';
            return;
        }
        role = role.trim();
    }

    if (!role) return;

    const cam = detectedCameras[cameraIndex];
    if (!currentProfile.data.cameras) currentProfile.data.cameras = {};

    const camId = String(cam.id || '');

    // Find the old role this physical camera was assigned to
    let oldRole = null;
    for (const [r, c] of Object.entries(currentProfile.data.cameras)) {
        if (String(c.index_or_path ?? c.serial_number_or_name ?? '') === camId && r !== role) {
            oldRole = r;
            break;
        }
    }

    // Find the physical camera currently occupying the target role
    const targetConfig = currentProfile.data.cameras[role] || null;
    const targetId = targetConfig ? String(targetConfig.index_or_path ?? targetConfig.serial_number_or_name ?? '') : '';

    // Auto-swap: if both cameras had roles, move the displaced camera to the old role
    if (oldRole && targetConfig && targetId) {
        // Swap: put target role's old camera into this camera's old role,
        // preserving the old role's settings (width, height, fps, fourcc)
        const oldRoleConfig = { ...currentProfile.data.cameras[oldRole] };
        // Update only the device identity to point to the displaced camera
        if (targetConfig.type === 'opencv') {
            oldRoleConfig.type = 'opencv';
            oldRoleConfig.index_or_path = targetConfig.index_or_path;
            delete oldRoleConfig.serial_number_or_name;
        } else if (targetConfig.type === 'intelrealsense') {
            oldRoleConfig.type = 'intelrealsense';
            oldRoleConfig.serial_number_or_name = targetConfig.serial_number_or_name;
            delete oldRoleConfig.index_or_path;
        }
        currentProfile.data.cameras[oldRole] = oldRoleConfig;
    } else if (oldRole) {
        // No swap target — just remove old role
        delete currentProfile.data.cameras[oldRole];
    }

    // Assign this camera to the target role, preserving target role's settings
    const existing = targetConfig || {};
    const camConfig = { ...existing };
    if (cam.type === 'OpenCV') {
        camConfig.type = 'opencv';
        camConfig.index_or_path = cam.id;
        delete camConfig.serial_number_or_name;
    } else if (cam.type === 'RealSense') {
        camConfig.type = 'intelrealsense';
        camConfig.serial_number_or_name = String(cam.id);
        delete camConfig.index_or_path;
    }
    // Only fill in settings from detected defaults if not already configured
    const profile = cam.default_stream_profile || {};
    if (!camConfig.width && profile.width) camConfig.width = profile.width;
    if (!camConfig.height && profile.height) camConfig.height = profile.height;
    if (!camConfig.fps && profile.fps) camConfig.fps = profile.fps;
    if (!camConfig.fourcc && profile.fourcc) camConfig.fourcc = profile.fourcc;

    currentProfile.data.cameras[role] = camConfig;
    _rerender();
    _updateDirtyState();
    const msg = oldRole && targetId
        ? `Swapped: ${cam.name || 'Camera'} → "${role}", displaced camera → "${oldRole}"`
        : `${cam.name || 'Camera'} assigned as "${role}"`;
    showToast('Camera assigned', msg, 'success');
}

// Re-render the entire editor from currentProfile state.
// Re-populates camera preview and port list from cached scan data.
function _rerender() {
    renderEditor();
    if (detectedCameras.length > 0) renderCameraPreview();
    if (scannedPorts) renderPortList(scannedPorts.ports, scannedPorts.allAssignments);
}

function removeCameraRole(role) {
    if (!currentProfile?.data?.cameras) return;
    delete currentProfile.data.cameras[role];
    _rerender();
    _updateDirtyState();
}

function updateCameraConfig(role, key, value) {
    if (!currentProfile?.data?.cameras?.[role]) return;
    if (key === 'use_depth') {
        currentProfile.data.cameras[role][key] = value === 'true';
    } else if (key === 'fourcc') {
        if (value) currentProfile.data.cameras[role][key] = value;
        else delete currentProfile.data.cameras[role][key];
    } else {
        // width, height, fps — numeric
        const n = parseInt(value, 10);
        if (isNaN(n) || n <= 0) delete currentProfile.data.cameras[role][key];
        else currentProfile.data.cameras[role][key] = n;
    }
    _updateDirtyState();
}

// ============================================================================
// Port scanning and arm identification
// ============================================================================

function renderPortsSection() {
    let html = '<div class="port-section"><h3>Arm Ports</h3>';
    html += '<div style="display: flex; gap: 8px; margin-bottom: 8px;">';
    html += '<button class="btn-small" id="scan-ports-btn" onclick="scanPorts()">Scan Ports</button>';
    html += '</div>';
    html += '<div class="port-list" id="port-list"></div>';
    html += '</div>';
    return html;
}

async function scanPorts() {
    const btn = document.getElementById('scan-ports-btn');
    if (btn) { btn.textContent = 'Scanning...'; btn.disabled = true; }

    try {
        const [portsRes, assignRes] = await Promise.all([
            fetch('/api/robot/ports'),
            fetch('/api/robot/all-port-assignments'),
        ]);
        const ports = await portsRes.json();
        const allAssignments = await assignRes.json();
        scannedPorts = { ports, allAssignments };
        renderPortList(ports, allAssignments);
    } catch (e) {
        showToast('Error', `Port scan failed: ${e.message}`, 'error');
    } finally {
        if (btn) { btn.textContent = 'Scan Ports'; btn.disabled = false; }
    }
}

function _getPortFields() {
    if (!currentProfile) return [];
    const schemas = currentProfile.kind === 'robot' ? robotSchemas : teleopSchemas;
    const schema = schemas?.find(s => s.type_name === currentProfile.data.type);
    if (!schema) return [];
    return schema.fields.filter(f =>
        f.name.includes('port') && f.type_str.toLowerCase().includes('str')
    );
}

function renderPortList(ports, allAssignments) {
    const list = document.getElementById('port-list');
    if (!list) return;
    if (ports.length === 0) {
        list.innerHTML = '<div style="color: #666; font-size: 13px; padding: 8px;">No serial ports found</div>';
        return;
    }
    const portFields = _getPortFields();

    // Map of physical port -> field name in the CURRENT profile, so we can recognize
    // a shared setup (the same device wired to the same role in more than one profile).
    const curKind = currentProfile ? currentProfile.kind : null;
    const curFields = (currentProfile && currentProfile.data.fields) || {};
    const curPortToField = {};
    for (const f of portFields) {
        const v = curFields[f.name];
        if (typeof v === 'string' && v.trim()) curPortToField[v.trim()] = f.name;
    }

    // Build map of ports claimed by OTHER profiles: { "/dev/ttyACM0": { profile_name, profile_kind, field_name } }
    const usedByOthers = {};
    if (allAssignments) {
        for (const a of allAssignments) {
            // Skip assignments belonging to the currently edited profile
            if (currentProfile && a.profile_name === currentProfile.name && a.profile_kind === currentProfile.kind) {
                continue;
            }
            // Shared setup: this profile maps the same port to the same field of the same
            // kind. That's one physical device playing one role across profiles (e.g. "white"
            // and "white_pred" sharing arms) — agreement, not a conflict. Don't flag it.
            if (a.profile_kind === curKind && curPortToField[a.port] === a.field_name) {
                continue;
            }
            usedByOthers[a.port] = a;
        }
    }

    list.innerHTML = ports.map(p => {
        const meta = [p.name || ''];
        if (p.manufacturer) meta.push(p.manufacturer);
        if (p.vid_pid) meta.push(p.vid_pid);

        const claimedBy = usedByOthers[p.path];

        // Build assign dropdown + "in use by" trailing text (always 5 children for grid alignment)
        let assignHtml = '<span></span>';
        let claimedHtml = '<span></span>';
        if (portFields.length > 0) {
            const currentField = portFields.find(f => {
                const input = document.getElementById(`field-${f.name}`);
                return input && input.value === p.path;
            })?.name || '';
            if (claimedBy) {
                // Port is used by another profile — still allow reassign, but with confirmation
                assignHtml = `<select class="port-assign-select port-assign-claimed" onchange="assignPort('${esc(p.path)}', this.value, '${esc(claimedBy.profile_name)}', '${esc(claimedBy.profile_kind)}')">
                    <option value="">${esc(claimedBy.field_name)}</option>
                    ${portFields.map(f => `<option value="${esc(f.name)}">${esc(f.name)}</option>`).join('')}
                </select>`;
                claimedHtml = `<span class="port-claimed">in use by <a onclick="selectProfile('${esc(claimedBy.profile_kind)}', '${esc(claimedBy.profile_name)}')">${esc(claimedBy.profile_name)}</a></span>`;
            } else {
                const unassignOpt = currentField ? `<option value="__unassign__">-- clear --</option>` : '';
                assignHtml = `<select class="port-assign-select" onchange="if(this.value==='__unassign__'){unassignPort('${esc(p.path)}')}else{assignPort('${esc(p.path)}', this.value)}">
                    <option value="">-- assign --</option>
                    ${unassignOpt}
                    ${portFields.map(f => `<option value="${esc(f.name)}" ${currentField === f.name ? 'selected' : ''}>${esc(f.name)}</option>`).join('')}
                </select>`;
            }
        }

        return `<div class="port-item">
            <span class="port-path">${esc(p.path)}</span>
            <span class="port-name">${esc(meta.join(' | '))}</span>
            <button class="btn-small" onclick="identifyArm('${esc(p.path)}', this)">Wiggle</button>
            ${assignHtml}
            ${claimedHtml}
        </div>`;
    }).join('');
}

function assignPort(port, fieldName, claimedByProfile, claimedByKind) {
    // If port is claimed by another profile, confirm before reassigning
    if (claimedByProfile) {
        if (!confirm(`This port is currently in use by "${claimedByProfile}" (${claimedByKind}). Reassign it anyway?`)) {
            // Reset the dropdown to show the claimed field
            if (scannedPorts) renderPortList(scannedPorts.ports, scannedPorts.allAssignments);
            return;
        }
    }
    if (!fieldName) return;
    if (!currentProfile.data.fields) currentProfile.data.fields = {};
    currentProfile.data.fields[fieldName] = port;
    _rerender();
    _updateDirtyState();
    showToast('Port set', `${fieldName} = ${port}`, 'info');
}

function unassignPort(port) {
    if (!currentProfile || !currentProfile.data.fields) return;
    const portFields = _getPortFields();
    for (const f of portFields) {
        if (currentProfile.data.fields[f.name] === port) {
            delete currentProfile.data.fields[f.name];
        }
    }
    _rerender();
    _updateDirtyState();
    showToast('Port cleared', `${port} unassigned`, 'info');
}

async function identifyArm(port, btn) {
    if (btn) {
        btn.textContent = 'Wiggling...';
        btn.classList.add('wiggling');
        btn.disabled = true;
    }
    try {
        const res = await fetch('/api/robot/identify-arm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ port }),
        });
        const result = await res.json();
        if (result.status === 'ok') {
            showToast('Arm moved', `Motor wiggled on ${port}`, 'success');
        } else {
            showToast('Error', result.message || 'Identification failed', 'error');
        }
    } catch (e) {
        showToast('Error', `Arm identification failed: ${e.message}`, 'error');
    } finally {
        if (btn) {
            btn.textContent = 'Wiggle';
            btn.classList.remove('wiggling');
            btn.disabled = false;
        }
    }
}

// ============================================================================
// Rest position
// ============================================================================

let _restRecordingActive = false;

function renderRestPositionSection() {
    const restPos = currentProfile?.data?.rest_position || {};
    const hasRest = Object.keys(restPos).length > 0;

    let html = '<div class="rest-position-section"><h3>Rest Position</h3>';

    if (hasRest) {
        html += '<div class="rest-position-values">';
        for (const [key, val] of Object.entries(restPos)) {
            const label = key.replace('.pos', '');
            html += `<span class="rest-pos-entry"><span class="rest-pos-label">${esc(label)}</span><span class="rest-pos-value">${Number(val).toFixed(1)}</span></span>`;
        }
        html += '</div>';
    } else {
        html += '<div class="rest-position-empty">No rest position recorded. Click Record to start.</div>';
    }

    html += '<div class="rest-position-actions">';
    if (_restRecordingActive) {
        html += '<button class="btn-small" id="finish-rest-btn" onclick="finishRestRecording()">Done</button>';
        html += '<button class="btn-small secondary" id="cancel-rest-btn" onclick="cancelRestRecording()">Cancel</button>';
    } else {
        html += '<button class="btn-small" id="record-rest-btn" onclick="startRestRecording()">Record Rest Position</button>';
        html += `<button class="btn-small secondary" id="move-rest-btn" onclick="moveToRestPosition()" ${hasRest ? '' : 'disabled'}>Move to Rest</button>`;
        html += '<button class="btn-small secondary" id="recover-robot-btn" onclick="recoverRobot()" title="Unwedge a robot whose motors are stuck in overload protection. Arm goes limp on success.">Recover</button>';
        html += '<button class="btn-small danger" id="clear-rest-btn" onclick="clearRestPosition()" ' + (hasRest ? '' : 'disabled') + '>Clear</button>';
    }
    html += '</div>';
    html += '<div id="rest-position-status" class="rest-position-status"></div>';
    html += '</div>';
    return html;
}

async function startRestRecording() {
    if (!currentProfile) return;

    const btn = document.getElementById('record-rest-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Connecting...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Connecting and disabling torque...';

    try {
        const res = await fetch('/api/robot/start-rest-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ robot: currentProfile.data }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to start recording');
        }

        _restRecordingActive = true;
        _rerender();
        const newStatus = document.getElementById('rest-position-status');
        if (newStatus) newStatus.textContent = 'Torque disabled — move the robot to the desired rest pose, then click Done.';
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Error', e.message, 'error');
        if (btn) { btn.textContent = 'Record Rest Position'; btn.disabled = false; }
    }
}

async function finishRestRecording() {
    const btn = document.getElementById('finish-rest-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Reading...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Reading positions...';

    try {
        const res = await fetch('/api/robot/finish-rest-recording', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to finish recording');
        }
        const result = await res.json();

        currentProfile.data.rest_position = result.rest_position;
        await _saveRestPositionToProfile();

        _restRecordingActive = false;
        _rerender();
        showToast('Recorded', 'Rest position saved to profile', 'success');
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Error reading positions. You can retry or cancel.';
        showToast('Error', e.message, 'error');
        if (btn) { btn.textContent = 'Done'; btn.disabled = false; }
    }
}

async function cancelRestRecording() {
    try {
        await fetch('/api/robot/cancel-rest-recording', { method: 'POST' });
    } catch (e) { /* ignore */ }
    _restRecordingActive = false;
    _rerender();
}

async function moveToRestPosition() {
    if (!currentProfile) return;
    const restPos = currentProfile.data.rest_position;
    if (!restPos || Object.keys(restPos).length === 0) return;

    const btn = document.getElementById('move-rest-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Moving...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Connecting and moving to rest position...';

    try {
        const res = await fetch('/api/robot/move-to-rest-position', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                robot: currentProfile.data,
                rest_position: restPos,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to move to rest position');
        }
        if (statusEl) statusEl.textContent = '';
        showToast('Done', 'Robot is at rest position', 'success');
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Error', e.message, 'error');
    } finally {
        if (btn) { btn.textContent = 'Move to Rest'; btn.disabled = false; }
    }
}

function clearRestPosition() {
    if (!currentProfile) return;
    if (!confirm('Clear the saved rest position?')) return;
    currentProfile.data.rest_position = {};
    _rerender();
    _updateDirtyState();
}

// ============================================================================
// Safe trajectory (hand-recorded joint-space motion for open-loop replay)
// ============================================================================

let _trajectoryRecordingActive = false;

function renderSafeTrajectorySection() {
    const restPos = currentProfile?.data?.rest_position || {};
    const hasRest = Object.keys(restPos).length > 0;
    const meta = currentProfile?._trajectoryMeta || { exists: false };

    let html = '<div class="rest-position-section"><h3>Safe Trajectory</h3>';

    if (meta.exists) {
        const dur = (meta.duration_s || 0).toFixed(1);
        const frames = meta.frame_count || 0;
        const fps = meta.fps || '?';
        html += `<div class="rest-position-values"><span class="rest-pos-entry"><span class="rest-pos-label">${frames} frames</span><span class="rest-pos-value">${dur}s @ ${fps} fps</span></span></div>`;
    } else {
        html += '<div class="rest-position-empty">No safe trajectory recorded. Records a hand-guided motion the robot can replay open-loop on its own.</div>';
    }

    html += '<div class="rest-position-actions">';
    if (_trajectoryRecordingActive) {
        html += '<button class="btn-small" id="stop-traj-btn" onclick="stopTrajectoryRecording()">Done</button>';
        html += '<button class="btn-small secondary" id="cancel-traj-btn" onclick="cancelTrajectoryRecording()">Cancel</button>';
    } else {
        const recordDisabled = hasRest ? '' : 'disabled';
        const recordTitle = hasRest
            ? 'title="Connect, move to rest, then disable torque. You hand-guide the arm; click Done to save."'
            : 'title="Record a rest position first — the trajectory must start from a known pose."';
        html += `<button class="btn-small" id="record-traj-btn" onclick="startTrajectoryRecording()" ${recordDisabled} ${recordTitle}>Record Trajectory</button>`;
        const replayDisabled = (meta.exists && hasRest) ? '' : 'disabled';
        const replayTitle = !hasRest
            ? 'title="Record a rest position first."'
            : !meta.exists ? 'title="Record a trajectory first."'
            : 'title="Move to rest, then replay the saved trajectory open-loop."';
        html += `<button class="btn-small secondary" id="replay-traj-btn" onclick="replayTrajectory()" ${replayDisabled} ${replayTitle}>Replay</button>`;
        const clearDisabled = meta.exists ? '' : 'disabled';
        html += `<button class="btn-small danger" id="clear-traj-btn" onclick="clearTrajectory()" ${clearDisabled}>Clear</button>`;
    }
    html += '</div>';
    html += '<div id="trajectory-status" class="rest-position-status"></div>';
    html += '</div>';
    return html;
}

async function startTrajectoryRecording() {
    if (!currentProfile) return;
    const restPos = currentProfile.data.rest_position;
    if (!restPos || Object.keys(restPos).length === 0) {
        showToast('Error', 'Record a rest position first', 'error');
        return;
    }

    const btn = document.getElementById('record-traj-btn');
    const statusEl = document.getElementById('trajectory-status');
    if (btn) { btn.textContent = 'Connecting...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Connecting, moving to rest, then disabling torque...';

    try {
        const res = await fetch('/api/robot/start-trajectory-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ robot: currentProfile.data, rest_position: restPos, fps: 30 }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to start recording');
        }
        _trajectoryRecordingActive = true;
        _rerender();
        const s = document.getElementById('trajectory-status');
        if (s) s.textContent = 'Torque disabled — hand-guide the arm through the safe motion, then click Done.';
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Error', e.message, 'error');
        if (btn) { btn.textContent = 'Record Trajectory'; btn.disabled = false; }
    }
}

async function stopTrajectoryRecording() {
    if (!currentProfile) return;
    const btn = document.getElementById('stop-traj-btn');
    const statusEl = document.getElementById('trajectory-status');
    if (btn) { btn.textContent = 'Saving...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Stopping sampler and saving trajectory...';

    try {
        const res = await fetch('/api/robot/stop-trajectory-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile_name: currentProfile.name }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to save trajectory');
        }
        const result = await res.json();
        _trajectoryRecordingActive = false;
        await _refreshTrajectoryMeta();
        showToast('Recorded', `${result.frame_count} frames over ${(result.duration_s || 0).toFixed(1)}s`, 'success');
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Error saving trajectory. You can retry or cancel.';
        showToast('Error', e.message, 'error');
        if (btn) { btn.textContent = 'Done'; btn.disabled = false; }
    }
}

async function cancelTrajectoryRecording() {
    try {
        await fetch('/api/robot/cancel-trajectory-recording', { method: 'POST' });
    } catch (e) { /* ignore */ }
    _trajectoryRecordingActive = false;
    _rerender();
}

async function replayTrajectory() {
    if (!currentProfile) return;
    const restPos = currentProfile.data.rest_position;
    if (!restPos || Object.keys(restPos).length === 0) {
        showToast('Error', 'Record a rest position first', 'error');
        return;
    }
    if (!confirm('Replay the saved trajectory? The robot will move to rest, then execute the recorded motion open-loop. Make sure no teleop session is using the port.')) return;

    const btn = document.getElementById('replay-traj-btn');
    const statusEl = document.getElementById('trajectory-status');
    if (btn) { btn.textContent = 'Replaying...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Connecting, moving to rest, then replaying...';

    try {
        const res = await fetch('/api/robot/replay-trajectory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ robot: currentProfile.data, rest_position: restPos, profile_name: currentProfile.name }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Replay failed');
        }
        if (statusEl) statusEl.textContent = '';
        showToast('Done', 'Replay complete', 'success');
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Error', e.message, 'error');
    } finally {
        if (btn) { btn.textContent = 'Replay'; btn.disabled = false; }
    }
}

async function clearTrajectory() {
    if (!currentProfile) return;
    if (!confirm('Delete the saved safe trajectory?')) return;
    try {
        await fetch(`/api/robot/trajectory/${encodeURIComponent(currentProfile.name)}`, { method: 'DELETE' });
    } catch (e) { /* ignore */ }
    await _refreshTrajectoryMeta();
}

async function recoverRobot() {
    /* Non-physical recovery of a wedged motor chain.
     *
     * Flow:
     *   1. Confirm with the user (the arm goes limp on success — gravity caveat).
     *   2. POST /api/robot/recover.
     *   3. If the report is fully clean (no still_unreachable_ids, no errors)
     *      AND a rest position is saved for this profile, automatically chain
     *      into POST /api/robot/move-to-rest-position. Use a longer
     *      duration_s when any motor was recovered from overload, so torque
     *      ramps gently and the just-recovered motor is less likely to
     *      re-trip mid-trajectory.
     *   4. Otherwise, surface the report and the gravity caveat.
     *
     * Pre: caller must ensure no active teleop / record / replay session is
     * holding the robot's serial port(s). Recovery opens each bus directly,
     * bypassing the strict handshake, so it must own the port.
     */
    if (!currentProfile) return;

    const restPos = currentProfile.data.rest_position;
    const hasRest = restPos && Object.keys(restPos).length > 0;

    const confirmMsg = hasRest
        ? 'Attempt non-physical recovery? On success the arm will be moved to the saved rest position. Make sure no teleop/record session is using the port.'
        : 'Attempt non-physical recovery? The arm will go LIMP on success — support it physically before clicking. Make sure no teleop/record session is using the port.';
    if (!confirm(confirmMsg)) return;

    const btn = document.getElementById('recover-robot-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Recovering...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Opening buses and pinging motors...';

    let reports;
    try {
        const res = await fetch('/api/robot/recover', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ robot: currentProfile.data }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Recovery failed');
        }
        const result = await res.json();
        reports = result.reports || [];
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Recovery failed', e.message, 'error');
        if (btn) { btn.textContent = 'Recover'; btn.disabled = false; }
        return;
    }

    // Aggregate across buses
    const stillUnreachable = [];
    const recovered = [];
    const errored = [];
    const noteLines = [];
    for (const r of reports) {
        for (const id of r.still_unreachable_ids || []) stillUnreachable.push(`${r.port || '?'}:${id}`);
        for (const id of r.recovered_ids || []) recovered.push(`${r.port || '?'}:${id}`);
        for (const [id, msg] of Object.entries(r.errors || {})) errored.push(`${r.port || '?'}:${id} (${msg})`);
        for (const n of r.notes || []) noteLines.push(`${r.port || '?'}: ${n}`);
    }

    // The backend emits a "no MotorsBus instances found on <RobotName>" note
    // when robot.recover_robot() found no SerialMotorsBus instances to act on.
    // That covers both legitimately-unsupported robots (Reachy2 over gRPC,
    // sim envs) and robot subclasses that store buses in non-standard places
    // (lists/dicts/lazy properties) where __dict__ introspection misses them.
    // Either way, recovery did *not* touch any motor — surface it as
    // unsupported instead of toasting "Recovered, arm is limp" (which would
    // be a lie).
    const unsupported = reports.some(
        r => (r.notes || []).some(n => n.includes('no MotorsBus')),
    );
    if (unsupported) {
        const robotType = currentProfile?.data?.type || 'this robot';
        const msg = `Recovery is not supported for "${robotType}" — no serial motor buses to act on.`;
        if (statusEl) statusEl.textContent = msg;
        showToast('Not supported', msg, 'error');
        if (btn) { btn.textContent = 'Recover'; btn.disabled = false; }
        return;
    }

    const clean = stillUnreachable.length === 0 && errored.length === 0;

    if (!clean) {
        // Report problems and stop — don't auto-move when state is uncertain.
        const lines = [];
        if (stillUnreachable.length) lines.push(`Still unreachable (power-cycle): ${stillUnreachable.join(', ')}`);
        if (errored.length) lines.push(`Errors: ${errored.join('; ')}`);
        if (noteLines.length) lines.push(noteLines.join('; '));
        if (statusEl) statusEl.textContent = lines.join(' — ');
        showToast('Recovery incomplete', stillUnreachable.length ? 'Some motors need a power cycle' : 'See status for details', 'error');
        if (btn) { btn.textContent = 'Recover'; btn.disabled = false; }
        return;
    }

    // Clean recovery — chain into Move-to-Rest if a rest position exists.
    if (!hasRest) {
        const summary = recovered.length
            ? `Recovered ${recovered.join(', ')}; arm is limp. Support it and re-pose.`
            : 'Recovery complete; arm is limp. Support it and re-pose.';
        if (statusEl) statusEl.textContent = summary;
        showToast('Recovered', 'Arm is limp — support it physically', 'success');
        if (btn) { btn.textContent = 'Recover'; btn.disabled = false; }
        return;
    }

    // Auto-chain Move-to-Rest. Double duration when anything was recovered to
    // ramp torque gently and reduce re-trip likelihood on the recovered motor.
    const duration_s = recovered.length ? 6.0 : 3.0;
    if (statusEl) statusEl.textContent = `Recovered. Moving to rest position (${duration_s}s)...`;

    try {
        const res = await fetch('/api/robot/move-to-rest-position', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                robot: currentProfile.data,
                rest_position: restPos,
                duration_s,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Move to rest failed after recovery');
        }
        if (statusEl) statusEl.textContent = '';
        const msg = recovered.length
            ? `Recovered ${recovered.join(', ')} and returned to rest position.`
            : 'Recovered and returned to rest position.';
        showToast('Done', msg, 'success');
    } catch (e) {
        // Recovery worked but move-to-rest didn't — arm is still limp.
        if (statusEl) statusEl.textContent = `Recovered, but move-to-rest failed: ${e.message}. Arm is limp — support it.`;
        showToast('Partial', 'Recovered but move-to-rest failed — arm is limp', 'error');
    } finally {
        if (btn) { btn.textContent = 'Recover'; btn.disabled = false; }
    }
}

async function _saveRestPositionToProfile() {
    /* Persist the current rest_position to the profile JSON on disk. */
    if (!currentProfile) return;
    const payload = {
        ...currentProfile.data,
        name: currentProfile.name,
        fields: _collectFormFields(),
    };
    try {
        const res = await fetch(`/api/robot/profiles/${encodeURIComponent(currentProfile.name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        currentProfile.data = payload;
        savedProfileData = JSON.parse(JSON.stringify(payload));
        _updateDirtyState();
    } catch (e) {
        console.error('Failed to auto-save rest position:', e);
    }
}

// ============================================================================
// Utilities
// ============================================================================

function esc(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}
