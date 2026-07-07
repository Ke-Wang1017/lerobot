// Bridge — receives host-driven UI commands from the MCP daemon.
//
// On page load this opens a WebSocket to /api/bridge/ws and listens for
// commands of type 'navigate', 'notify', 'highlight', and 'filter'.
//
// The hash route convention is the user-visible deep-link form:
//
//   #/home
//   #/dataset/<encoded repo_id>
//   #/episode/<encoded repo_id>/<episode_id>
//   #/run/<run_id>
//
// hashchange fires both for backend-driven commands (we set
// location.hash) and for manually clicked deep-links. One handler
// covers both paths.
//
// Decision #9 (architectural plan): every command carries `client_id`.
// The tab subscribes with `as=*` by default; a future UI affordance can
// set `lerobot_as=<name>` cookie to scope this tab to one user.

"use strict";

(function () {
  const LOG_PREFIX = "[lerobot-bridge]";
  const ORIGINAL_TITLE = document.title;
  let titleFlashTimer = null;
  let ws = null;
  let wsReconnectDelayMs = 1000;

  // ── Hash routing ──────────────────────────────────────────────────────

  // Tabs declared by index.html. We surface this here so the bridge
  // can request tab switches without knowing how the rest of the SPA
  // is structured.
  const VIEW_TO_TAB = {
    home: "data",
    dataset: "data",
    episode: "data",
    run: "run",
    model: "model",
    robot: "robot",
  };

  function switchTabIfPresent(tabName) {
    if (typeof window.switchTab === "function") {
      try {
        window.switchTab(tabName);
      } catch (e) {
        console.warn(LOG_PREFIX, "switchTab failed:", e);
      }
    }
  }

  function encodeRepoId(repoId) {
    // repo_ids like "org/dataset" must survive a hash path; encode each
    // segment so "/" inside the path stays literal.
    return String(repoId)
      .split("/")
      .map(encodeURIComponent)
      .join("/");
  }

  // Build the deeplink string for a navigation. Pure; testable.
  function deepLinkFor(view, params) {
    params = params || {};
    if (view === "episode" && params.repo_id != null && params.episode_id != null) {
      return `#/episode/${encodeRepoId(params.repo_id)}/${Number(params.episode_id)}`;
    }
    if (view === "dataset" && params.repo_id != null) {
      return `#/dataset/${encodeRepoId(params.repo_id)}`;
    }
    if (view === "run" && params.run_id != null) {
      return `#/run/${encodeURIComponent(params.run_id)}`;
    }
    if (view === "home") return "#/home";
    if (view === "model") return "#/model";
    if (view === "robot") return "#/robot";
    return null;
  }

  // Parse a hash like "#/episode/foo%2Fbar/47" into a route object.
  function parseHash(hash) {
    if (!hash || !hash.startsWith("#/")) return null;
    const parts = hash.slice(2).split("/").filter((s) => s !== "");
    if (parts.length === 0) return null;
    const head = parts[0];
    if (head === "home" || head === "model" || head === "robot") {
      return { view: head, params: {} };
    }
    if (head === "dataset" && parts.length >= 3) {
      return { view: "dataset", params: { repo_id: `${decodeURIComponent(parts[1])}/${decodeURIComponent(parts[2])}` } };
    }
    if (head === "episode" && parts.length >= 4) {
      return {
        view: "episode",
        params: {
          repo_id: `${decodeURIComponent(parts[1])}/${decodeURIComponent(parts[2])}`,
          episode_id: Number(parts[3]),
        },
      };
    }
    if (head === "run" && parts.length >= 2) {
      return { view: "run", params: { run_id: decodeURIComponent(parts[1]) } };
    }
    return null;
  }

  // Apply a route — switch tab, dispatch a custom DOM event so other
  // GUI code can react. The route is also reflected in the URL hash.
  function applyRoute(view, params) {
    const tab = VIEW_TO_TAB[view];
    if (tab) switchTabIfPresent(tab);
    // Broadcast a custom event other modules can subscribe to.
    // Keeps the bridge decoupled from the rest of the GUI.
    document.dispatchEvent(
      new CustomEvent("lerobot-bridge:navigate", { detail: { view, params: params || {} } }),
    );
  }

  function onHashChange() {
    const route = parseHash(window.location.hash);
    if (route) applyRoute(route.view, route.params);
  }

  // Apply the URL hash on first load so deep-links work directly. bridge.js
  // is loaded near the end of <body> (so it might run before or after
  // DOMContentLoaded depending on script ordering / `defer`); fire the handler
  // immediately if the document is already parsed, otherwise wait for it.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onHashChange);
  } else {
    onHashChange();
  }
  window.addEventListener("hashchange", onHashChange);

  // ── Notifications ─────────────────────────────────────────────────────

  function flashTitle(text) {
    // Cycle the title between an alert and the original until the user
    // refocuses the tab. Works without notification permission.
    if (titleFlashTimer) clearInterval(titleFlashTimer);
    let on = true;
    titleFlashTimer = setInterval(() => {
      document.title = on ? `🔔 ${text}` : ORIGINAL_TITLE;
      on = !on;
    }, 1000);
    const stop = () => {
      if (titleFlashTimer) clearInterval(titleFlashTimer);
      titleFlashTimer = null;
      document.title = ORIGINAL_TITLE;
      window.removeEventListener("focus", stop);
    };
    window.addEventListener("focus", stop);
  }

  async function ensureNotificationPermission() {
    if (!("Notification" in window)) return "unsupported";
    if (Notification.permission === "granted" || Notification.permission === "denied") {
      return Notification.permission;
    }
    try {
      return await Notification.requestPermission();
    } catch {
      return "denied";
    }
  }

  async function showNotification(params) {
    const title = params.title || "LeRobot";
    const body = params.body || "";
    const deeplink = params.deeplink || null;

    const permission = await ensureNotificationPermission();
    if (permission !== "granted") {
      flashTitle(title);
      return;
    }
    try {
      const n = new Notification(title, { body, tag: "lerobot-bridge" });
      n.onclick = (ev) => {
        ev.preventDefault();
        try {
          window.focus();
        } catch {}
        if (deeplink) {
          // hashchange fires automatically when we assign to location.hash;
          // route + tab follow.
          window.location.hash = deeplink.startsWith("#") ? deeplink : `#${deeplink}`;
        }
        n.close();
      };
    } catch (e) {
      console.warn(LOG_PREFIX, "Notification failed; falling back to title flash", e);
      flashTitle(title);
    }
  }

  // ── Command handlers ──────────────────────────────────────────────────

  function handleNavigate(params) {
    const link = deepLinkFor(params.view, params.params);
    if (link) {
      // Setting hash also triggers hashchange → applyRoute.
      window.location.hash = link;
    } else {
      console.warn(LOG_PREFIX, "navigate: unknown view shape", params);
    }
  }

  function handleNotify(params) {
    showNotification(params || {});
  }

  function handleHighlight(params) {
    // V1 placeholder — the existing GUI doesn't yet expose a
    // highlight surface. Broadcast as a custom event so a future
    // listing component can subscribe.
    document.dispatchEvent(
      new CustomEvent("lerobot-bridge:highlight", { detail: params || {} }),
    );
    console.log(LOG_PREFIX, "highlight", params);
  }

  function handleFilter(params) {
    document.dispatchEvent(
      new CustomEvent("lerobot-bridge:filter", { detail: params || {} }),
    );
    console.log(LOG_PREFIX, "filter", params);
  }

  function handleHello(msg) {
    console.log(LOG_PREFIX, "connected; target=", msg.target);
    document.dispatchEvent(
      new CustomEvent("lerobot-bridge:hello", { detail: { target: msg.target } }),
    );
  }

  function dispatchCommand(msg) {
    if (!msg || typeof msg !== "object") return;
    switch (msg.type) {
      case "hello":
        handleHello(msg);
        return;
      case "navigate":
        handleNavigate(msg.params || {});
        return;
      case "notify":
        handleNotify(msg.params || {});
        return;
      case "highlight":
        handleHighlight(msg.params || {});
        return;
      case "filter":
        handleFilter(msg.params || {});
        return;
      default:
        console.log(LOG_PREFIX, "ignored command", msg);
    }
  }

  // ── WebSocket lifecycle ──────────────────────────────────────────────

  function readCookie(name) {
    const all = document.cookie.split(";").map((s) => s.trim());
    for (const c of all) {
      const i = c.indexOf("=");
      if (i > 0 && c.slice(0, i) === name) return decodeURIComponent(c.slice(i + 1));
    }
    return null;
  }

  function wsUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    // Cookie `lerobot_as=<client_id>` opts the tab into scoped delivery;
    // otherwise we declare wildcard.
    const target = readCookie("lerobot_as") || "*";
    return `${proto}//${window.location.host}/api/bridge/ws?as=${encodeURIComponent(target)}`;
  }

  function connect() {
    const url = wsUrl();
    try {
      ws = new WebSocket(url);
    } catch (e) {
      console.warn(LOG_PREFIX, "ws construct failed:", e);
      scheduleReconnect();
      return;
    }
    ws.addEventListener("open", () => {
      console.log(LOG_PREFIX, "ws open", url);
      wsReconnectDelayMs = 1000;
    });
    ws.addEventListener("message", (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      dispatchCommand(msg);
    });
    ws.addEventListener("close", () => {
      console.log(LOG_PREFIX, "ws closed; reconnecting");
      scheduleReconnect();
    });
    ws.addEventListener("error", (e) => {
      console.warn(LOG_PREFIX, "ws error:", e);
    });
  }

  function scheduleReconnect() {
    setTimeout(connect, wsReconnectDelayMs);
    // Exponential backoff capped at 30s.
    wsReconnectDelayMs = Math.min(wsReconnectDelayMs * 2, 30000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", connect);
  } else {
    connect();
  }

  // Expose a tiny test/debug surface — useful for autonomous tests and
  // for the browser console.
  window.lerobotBridge = {
    parseHash,
    deepLinkFor,
    dispatchCommand,
    isConnected: () => ws && ws.readyState === WebSocket.OPEN,
  };
})();
