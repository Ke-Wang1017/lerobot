// Copyright 2026 The HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// charts.js — the ONE line-chart primitive for the GUI.
//
// Supersedes the three overlapping canvas charters that used to live in
// run.js (_drawSparkline, _drawSparklineMulti [dead], _registerSyncedChart),
// which duplicated ~80% of their drawing and split hover support unevenly
// (only the synced one had it). One renderer, hover everywhere.
//
// Key difference from the old _registerSyncedChart: hover/crosshair state is
// scoped to a *sync group*, not app-wide module globals — so independent
// panels (training, RLT, performance) can each render charts without
// polluting each other's shared X-axis or crosshair.
//
// Usage:
//   drawChart('my-canvas', { series: [{ data:[...], color:'#34d399', label:'loss' }] });
//   // synced crosshair across several charts: give them the same group
//   drawChart('a', { series, syncGroup: 'rlt', latestStep: 1200, timestamps });
//   drawChart('b', { series, syncGroup: 'rlt', latestStep: 1200, timestamps });
//
// Options:
//   series:     [{ data:[Number], color, label?, percentage?, hideLine?,
//                  bandPair?:'min'|'max', bandColor? }]
//   syncGroup:  string — charts sharing it share a crosshair + X axis (N).
//               Defaults to the canvasId (a group of one).
//   fixedMin/fixedMax: lock the Y range (else auto from data).
//   percentage: format values/labels as N% (per-chart default; series can override).
//   timestamps: [unixSeconds] aligned to the longest series, for the hover label.
//   latestStep: global step of the newest point, for the hover "step N" label.

const _chartGroups = {}; // group name -> { hoverIndex, latestStep, charts: {canvasId: spec} }

function _chartGroup(name) {
  if (!_chartGroups[name]) _chartGroups[name] = { hoverIndex: -1, latestStep: 0, charts: {} };
  return _chartGroups[name];
}

function _chartFmtTime(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString();
}

function _chartFmtAgo(ts) {
  if (!ts) return "";
  const ago = Math.max(0, Date.now() / 1000 - ts);
  if (ago < 60) return Math.round(ago) + "s ago";
  if (ago < 3600) return Math.round(ago / 60) + "m ago";
  return Math.round(ago / 3600) + "h ago";
}

// Value label: scientific for very small/large magnitudes (a learning rate of
// 1e-5 must not render as "0.0000"), fixed otherwise.
function _chartFmtValue(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(1);
  return a < 1 ? v.toFixed(4) : v.toFixed(3);
}

// Longest series across all charts in a group → shared X axis. Charts are
// right-aligned: a short series occupies the right portion of the canvas.
function _chartGroupN(group) {
  let maxN = 1;
  for (const spec of Object.values(group.charts)) {
    for (const s of spec.series) {
      if (s.data && s.data.length > maxN) maxN = s.data.length;
    }
    if (spec.timestamps && spec.timestamps.length > maxN) maxN = spec.timestamps.length;
  }
  return maxN;
}

// Public entry point. Registers/updates the chart in its group and (re)draws
// every chart in that group so a shared crosshair stays in sync.
function drawChart(canvasId, opts) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const series = (opts.series || []).filter((s) => s.data && s.data.length > 0);
  if (series.length === 0) return;

  const groupName = opts.syncGroup || canvasId;
  const group = _chartGroup(groupName);
  if (opts.latestStep != null) group.latestStep = opts.latestStep;

  // Size the backing store for crisp lines on HiDPI.
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.getContext("2d").scale(dpr, dpr);

  // logY keeps a wide-range series (a loss going 40 → 0.1 would otherwise let
  // the warm-up spike flatten the whole curve) readable. Range is in log space;
  // values still display linearly.
  const logY = !!opts.logY;
  const proj = (v) => (logY ? Math.log10(Math.max(v, 1e-9)) : v);
  let allVals = [];
  for (const s of series) allVals = allVals.concat(s.data);
  const min = opts.fixedMin !== undefined ? opts.fixedMin : Math.min(...allVals.map(proj));
  const max = opts.fixedMax !== undefined ? opts.fixedMax : Math.max(...allVals.map(proj));

  group.charts[canvasId] = {
    canvas,
    groupName,
    series,
    timestamps: opts.timestamps || [],
    percentage: !!opts.percentage,
    logY,
    min,
    max,
    pad: 4,
    W: rect.width,
    H: rect.height,
  };

  // Attach hover handlers once per canvas. They mutate the GROUP's hoverIndex
  // and redraw the group — no app-wide state, so other groups are untouched.
  if (!canvas._chartHoverAttached) {
    canvas.addEventListener("mousemove", (e) => {
      const spec = group.charts[canvasId];
      if (!spec) return;
      const x = e.clientX - canvas.getBoundingClientRect().left;
      const N = _chartGroupN(group);
      group.hoverIndex = Math.min(N - 1, Math.max(0, Math.round((x / spec.W) * (N - 1))));
      _redrawGroup(group);
    });
    canvas.addEventListener("mouseleave", () => {
      group.hoverIndex = -1;
      _redrawGroup(group);
    });
    canvas._chartHoverAttached = true;
  }

  _redrawGroup(group);
}

function _redrawGroup(group) {
  for (const id of Object.keys(group.charts)) _renderChart(group.charts[id]);
}

function _renderChart(spec) {
  const { canvas, series, W, H, pad, min, max } = spec;
  const group = _chartGroups[spec.groupName];
  const hoverIndex = group ? group.hoverIndex : -1;
  const ctx = canvas.getContext("2d");
  const range = max - min || 1;
  const N = group ? _chartGroupN(group) : Math.max(...series.map((s) => s.data.length), 1);

  ctx.clearRect(0, 0, W, H);

  // Horizontal gridlines.
  ctx.strokeStyle = "#1a1a3e";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (H - 2 * pad) * (1 - i / 4);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  // Zero line when the range straddles zero.
  if (!spec.logY && min < 0 && max > 0) {
    const zeroY = pad + (H - 2 * pad) * (1 - (0 - min) / range);
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(W, zeroY);
    ctx.stroke();
  }

  const projY = (v) => (spec.logY ? Math.log10(Math.max(v, 1e-9)) : v);
  const toY = (v) => pad + (H - 2 * pad) * (1 - (projY(v) - min) / range);
  const toXGlobal = (i) => (i / Math.max(N - 1, 1)) * W;
  const globalIdx = (s, localIdx) => N - s.data.length + localIdx;

  // Optional min/max band fill (a pair of series tagged bandPair).
  const bandMin = series.find((s) => s.bandPair === "min");
  const bandMax = series.find((s) => s.bandPair === "max");
  if (bandMin && bandMax && bandMin.data.length > 0 && bandMax.data.length > 0) {
    ctx.fillStyle = (bandMin.bandColor || bandMin.color) + "33";
    ctx.beginPath();
    for (let i = 0; i < bandMax.data.length; i++) ctx.lineTo(toXGlobal(globalIdx(bandMax, i)), toY(bandMax.data[i]));
    for (let i = bandMin.data.length - 1; i >= 0; i--) ctx.lineTo(toXGlobal(globalIdx(bandMin, i)), toY(bandMin.data[i]));
    ctx.closePath();
    ctx.fill();
  }

  // Series lines.
  for (const s of series) {
    if (s.data.length === 0 || s.hideLine) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < s.data.length; i++) {
      const x = toXGlobal(globalIdx(s, i));
      const y = toY(s.data[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Hover crosshair.
  if (hoverIndex >= 0 && hoverIndex < N) {
    const x = toXGlobal(hoverIndex);
    ctx.strokeStyle = "#888";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Value labels (top-right) — value at the hovered index, else latest.
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  for (let i = 0; i < series.length; i++) {
    const s = series[i];
    if (s.data.length === 0) continue;
    let localIdx;
    if (hoverIndex >= 0) {
      localIdx = hoverIndex - (N - s.data.length);
      if (localIdx < 0 || localIdx >= s.data.length) {
        ctx.fillStyle = s.color;
        ctx.fillText("—", W - 4, 12 + i * 12);
        continue;
      }
    } else {
      localIdx = s.data.length - 1;
    }
    const v = s.data[localIdx];
    const pct = s.percentage != null ? s.percentage : spec.percentage;
    ctx.fillStyle = s.color;
    ctx.fillText(pct ? (v * 100).toFixed(0) + "%" : _chartFmtValue(v), W - 4, 12 + i * 12);
  }

  // Step / time label (bottom-left) — this chart's own timestamps.
  ctx.fillStyle = "#444";
  ctx.textAlign = "left";
  if (hoverIndex >= 0) {
    const latestStep = group ? group.latestStep : 0;
    const globalStep = latestStep - (N - 1 - hoverIndex);
    const ts = spec.timestamps;
    const tsLocalIdx = hoverIndex - (N - ts.length);
    const t = tsLocalIdx >= 0 && tsLocalIdx < ts.length ? ts[tsLocalIdx] : null;
    ctx.fillText(t ? `step ${globalStep}  ${_chartFmtTime(t)} (${_chartFmtAgo(t)})` : `step ${globalStep}`, 4, H - 4);
  } else {
    ctx.fillText(N + " pts", 4, H - 4);
  }
}

// Drop a sync group's charts from the registry (call when a panel's view is
// torn down) so a stale chart can't widen another panel's shared X axis.
function clearChartGroup(name) {
  delete _chartGroups[name];
}
