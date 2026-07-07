// Bug report dialog. Captures a DOM screenshot via html2canvas (lazy-loaded
// from CDN), submits to /api/bug_reports, and shows a toast pointing the user
// at the saved directory. Screenshot capture degrades gracefully if the CDN
// is unreachable — the report is still saved without the image.

const HTML2CANVAS_CDN = 'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js';

let _html2canvasLoaded = null; // Promise once we've started loading

function _loadHtml2Canvas() {
    if (_html2canvasLoaded) return _html2canvasLoaded;
    _html2canvasLoaded = new Promise((resolve, reject) => {
        if (typeof window.html2canvas === 'function') {
            resolve(window.html2canvas);
            return;
        }
        const s = document.createElement('script');
        s.src = HTML2CANVAS_CDN;
        s.async = true;
        s.onload = () => {
            if (typeof window.html2canvas === 'function') {
                resolve(window.html2canvas);
            } else {
                reject(new Error('html2canvas loaded but symbol missing'));
            }
        };
        s.onerror = () => reject(new Error('failed to load html2canvas from CDN'));
        document.head.appendChild(s);
    });
    return _html2canvasLoaded;
}

function openBugReportDialog() {
    const overlay = document.getElementById('bug-report-overlay');
    document.getElementById('bug-report-title').value = '';
    document.getElementById('bug-report-description').value = '';
    document.getElementById('bug-report-include-screenshot').checked = true;
    document.getElementById('bug-report-status').textContent = '';
    document.getElementById('bug-report-submit').disabled = false;
    overlay.style.display = 'flex';
    document.getElementById('bug-report-title').focus();
}

function closeBugReportDialog() {
    document.getElementById('bug-report-overlay').style.display = 'none';
}

function _activeTabName() {
    const active = document.querySelector('.tab.active');
    return active ? (active.getAttribute('data-tab') || active.textContent.trim()) : null;
}

async function _captureScreenshot() {
    // Briefly hide the modal so it doesn't appear in the screenshot of the
    // "broken" view. We restore display immediately after the capture starts;
    // html2canvas works off the snapshot it took at call time.
    const overlay = document.getElementById('bug-report-overlay');
    const prevDisplay = overlay.style.display;
    overlay.style.display = 'none';
    try {
        const html2canvas = await _loadHtml2Canvas();
        const canvas = await html2canvas(document.body, {
            backgroundColor: '#16213e',
            useCORS: true,
            logging: false,
            // Cap dimensions: 4K screenshots are ~10 MB as PNG, which is fine,
            // but we don't need to render at device-pixel-ratio. CSS pixels are
            // plenty for a bug report.
            scale: 1,
        });
        return canvas.toDataURL('image/png');
    } finally {
        overlay.style.display = prevDisplay;
    }
}

async function submitBugReport() {
    const title = document.getElementById('bug-report-title').value.trim();
    const description = document.getElementById('bug-report-description').value;
    const includeScreenshot = document.getElementById('bug-report-include-screenshot').checked;
    const status = document.getElementById('bug-report-status');
    const submitBtn = document.getElementById('bug-report-submit');

    if (!title) {
        status.textContent = 'Please enter a title.';
        return;
    }

    submitBtn.disabled = true;
    let screenshotDataUrl = null;
    if (includeScreenshot) {
        status.textContent = 'Capturing screenshot...';
        try {
            screenshotDataUrl = await _captureScreenshot();
        } catch (e) {
            status.textContent = `Screenshot failed (${e.message}). Submitting without it...`;
        }
    }

    status.textContent = 'Saving report...';
    const payload = {
        title,
        description,
        url: window.location.href,
        user_agent: navigator.userAgent,
        viewport: {
            width: window.innerWidth,
            height: window.innerHeight,
            dpr: window.devicePixelRatio || 1,
        },
        active_tab: _activeTabName(),
        screenshot_data_url: screenshotDataUrl,
        client_extra: {},
    };

    try {
        const resp = await fetch('/api/bug_reports', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err}`);
        }
        const data = await resp.json();
        closeBugReportDialog();
        if (typeof showToast === 'function') {
            const shotMsg = data.screenshot_saved ? ' (with screenshot)' : '';
            showToast('Bug report saved', `${data.directory}${shotMsg}`, 'success', 8000);
        }
    } catch (e) {
        status.textContent = `Failed to submit: ${e.message}`;
        submitBtn.disabled = false;
    }
}

// Esc closes the dialog when it's open.
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const overlay = document.getElementById('bug-report-overlay');
    if (overlay && overlay.style.display === 'flex') {
        closeBugReportDialog();
    }
});

// Deep-link: visiting /#bug-report (or any URL ending in #bug-report) auto-opens
// the dialog after the page settles. Useful for bookmarks and for shareable
// "click here to file a bug" links.
if (window.location.hash === '#bug-report') {
    window.addEventListener('DOMContentLoaded', () => openBugReportDialog());
}
