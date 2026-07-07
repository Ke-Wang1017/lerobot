// Bridge consumers — handle CustomEvents fired by bridge.js, turning
// AI-driven MCP commands into visible UI changes.
//
// bridge.js is the transport (WebSocket → CustomEvent); this module is
// the consumer (CustomEvent → call into app.js / feature modules). The
// split keeps bridge.js feature-agnostic.

(() => {
  const LOG_PREFIX = "[bridge-consumer]";

  // ── Navigate: open a dataset / seek to an episode in the Data tab ──

  document.addEventListener("lerobot-bridge:navigate", (e) => {
    const { view, params } = e.detail || {};
    if (!view) return;

    // For dataset / episode, the tab switch already happened in bridge.js;
    // we additionally drive the data-tab inspector into the right state.
    if (view === "dataset" && params && params.repo_id) {
      _openDatasetThenMaybe(params.repo_id, null);
    } else if (view === "episode" && params && params.repo_id && params.episode_id != null) {
      _openDatasetThenMaybe(params.repo_id, Number(params.episode_id));
    }
  });

  async function _openDatasetThenMaybe(repoId, episodeId) {
    if (typeof window.openDataset !== "function") {
      console.warn(LOG_PREFIX, "openDataset not available; cannot navigate");
      return;
    }
    // If already opened, openDataset is idempotent enough — it'll re-fetch
    // and set active. If not opened, this fetches metadata + adds it.
    try {
      await window.openDataset(repoId);
    } catch (err) {
      console.warn(LOG_PREFIX, "openDataset failed:", err);
      return;
    }
    if (episodeId == null) return;

    // After the open completes, episodes should be populated. Pick the row
    // and call selectEpisode with its length.
    const ds = (window.datasets || window.openedDatasets || {})[repoId];
    if (!ds) {
      console.warn(LOG_PREFIX, "dataset not found in registry after open:", repoId);
      return;
    }
    const episodes = (window.episodes || {})[repoId];
    if (!episodes || !episodes[episodeId]) {
      console.warn(LOG_PREFIX, "episode row not yet loaded; deferring");
      return;
    }
    const length = episodes[episodeId].length;
    if (typeof window.selectEpisode === "function") {
      window.selectEpisode(repoId, Number(episodeId), Number(length));
    }
  }

  // ── Highlight: mark matching episode rows so the user notices them ──

  document.addEventListener("lerobot-bridge:highlight", (e) => {
    const { repo_id, episode_ids } = e.detail || {};
    if (!repo_id || !Array.isArray(episode_ids)) return;
    _applyHighlight(repo_id, episode_ids);
  });

  function _applyHighlight(repoId, episodeIds) {
    // Drive state, not DOM: bridgeHighlights is a Map<repo_id, Set<episode_id>>
    // that renderTree() reads when emitting rows. This way the highlight
    // survives the next renderTree() call (e.g. when selectEpisode marks a
    // different row active). DOM mutation directly was a race we kept losing.
    const state = window.bridgeHighlights;
    if (!state) {
      console.warn(LOG_PREFIX, "bridgeHighlights state missing; falling back to DOM mutation");
      _applyHighlightDom(repoId, episodeIds);
      return;
    }
    state.clear();  // "set" semantics — successive calls replace
    state.set(repoId, new Set(episodeIds.map(Number)));
    if (typeof window.renderTree === "function") {
      window.renderTree();
    }
    // After the re-render, scroll the first match into view.
    const sel =
      '[data-episode-row][data-dataset-id="' + repoId + '"][data-episode-id="' +
      Number(episodeIds[0]) + '"]';
    const el = document.querySelector(sel);
    if (el) {
      el.scrollIntoView({ block: "center", behavior: "instant" });
    } else {
      console.warn(LOG_PREFIX, "highlight: no rows matched", { repoId, episodeIds });
    }
  }

  function _applyHighlightDom(repoId, episodeIds) {
    document.querySelectorAll(".bridge-highlight").forEach((el) => {
      el.classList.remove("bridge-highlight");
    });
    const wanted = new Set(episodeIds.map((n) => String(n)));
    const rows = document.querySelectorAll('[data-episode-row][data-dataset-id]');
    let firstHit = null;
    rows.forEach((el) => {
      const ds = el.getAttribute("data-dataset-id");
      const ep = el.getAttribute("data-episode-id");
      if (ds === repoId && wanted.has(String(ep))) {
        el.classList.add("bridge-highlight");
        if (!firstHit) firstHit = el;
      }
    });
    if (firstHit) {
      firstHit.scrollIntoView({ block: "center", behavior: "instant" });
    }
  }

  // ── Filter: stub. No viewer has a real filter input today, so the
  //    listener is intentionally not wired — see bridge_tools.py and the
  //    "filter UX" TODO. When the data tab grows a search input, drop the
  //    handler back in here and re-register the set_filter MCP tool.

  // ── Tag indicators on episode rows ──
  //
  // When the AI tags an episode via `tag_episode`, we tell the user by
  // putting a small badge on the relevant row. The MCP tool doesn't push
  // an event to the GUI today, so we listen for the bridge command and
  // refresh the row's badge from the tags API. This is best-effort; a
  // proper "comments surfaced in GUI" feature is decision #7 in the plan.

  // (deferred — depends on bridge.js / MCP pushing a "tags-changed" event
  //  which isn't in the protocol yet. Filed under the comment-surface TODO.)
})();
