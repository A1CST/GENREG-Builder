// GENREG layout: draggable dividers (sidebars + terminal dock) and the main canvas.

// -- draggable dividers ----------------------------------------------------
const workspace = document.getElementById("upper");
const sidebarLeft = document.getElementById("sidebar-left");
const sidebarRight = document.getElementById("sidebar-right");
const terminalPanel = document.getElementById("terminal-panel");

const CLAMP = {
  sidebar: { min: 140, max: 560 },
  terminal: { min: 120, max: 0.85 },   // max as fraction of workspace height
};

function makeDrag(handle, axis, onMove) {
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    const start = axis === "x" ? e.clientX : e.clientY;
    const move = (ev) => onMove((axis === "x" ? ev.clientX : ev.clientY) - start, start);
    const up = (ev) => {
      handle.releasePointerCapture(e.pointerId);
      handle.classList.remove("dragging");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
    };
    // Snapshot starting sizes so deltas are absolute.
    onMove._base = null;
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
}

// Left sidebar width
(() => {
  const h = document.querySelector('.resizer.v[data-resize="left"]');
  let base = 0;
  h.addEventListener("pointerdown", () => (base = sidebarLeft.getBoundingClientRect().width));
  makeDrag(h, "x", (dx) => {
    const w = Math.min(CLAMP.sidebar.max, Math.max(CLAMP.sidebar.min, base + dx));
    sidebarLeft.style.width = w + "px";
  });
})();

// Right sidebar width (drag left = wider)
(() => {
  const h = document.querySelector('.resizer.v[data-resize="right"]');
  let base = 0;
  h.addEventListener("pointerdown", () => (base = sidebarRight.getBoundingClientRect().width));
  makeDrag(h, "x", (dx) => {
    const w = Math.min(CLAMP.sidebar.max, Math.max(CLAMP.sidebar.min, base - dx));
    sidebarRight.style.width = w + "px";
  });
})();

// Terminal dock height (drag up = taller)
(() => {
  const h = document.getElementById("term-resizer");
  let base = 0;
  h.addEventListener("pointerdown", () => (base = terminalPanel.getBoundingClientRect().height));
  makeDrag(h, "y", (dy) => {
    const wsH = document.querySelector(".workspace").getBoundingClientRect().height;
    const max = wsH * CLAMP.terminal.max;
    const height = Math.min(max, Math.max(CLAMP.terminal.min, base - dy));
    terminalPanel.style.height = height + "px";
    resizeCanvas();   // canvas shares the vertical space
  });
})();

// -- layout persistence (cookies) ------------------------------------------
// Remember the panel sizes across refreshes so the user doesn't re-drag them.
const LAYOUT_COOKIE = "genreg_layout";
const clampN = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function setCookie(name, value) {
  const oneYear = 60 * 60 * 24 * 365;
  document.cookie = `${name}=${encodeURIComponent(value)};path=/;max-age=${oneYear};SameSite=Lax`;
}
function getCookie(name) {
  const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  return m ? decodeURIComponent(m[1]) : null;
}

function saveLayout() {
  setCookie(LAYOUT_COOKIE, JSON.stringify({
    left: Math.round(sidebarLeft.getBoundingClientRect().width),
    right: Math.round(sidebarRight.getBoundingClientRect().width),
    term: Math.round(terminalPanel.getBoundingClientRect().height),
  }));
}

function restoreLayout() {
  const raw = getCookie(LAYOUT_COOKIE);
  if (!raw) return;
  let d;
  try { d = JSON.parse(raw); } catch (_) { return; }
  if (d.left)  sidebarLeft.style.width  = clampN(d.left,  CLAMP.sidebar.min, CLAMP.sidebar.max) + "px";
  if (d.right) sidebarRight.style.width = clampN(d.right, CLAMP.sidebar.min, CLAMP.sidebar.max) + "px";
  if (d.term) {
    const wsH = document.querySelector(".workspace").getBoundingClientRect().height;
    terminalPanel.style.height = clampN(d.term, CLAMP.terminal.min, wsH * CLAMP.terminal.max) + "px";
  }
}

// Save whenever a drag on any resizer finishes.
for (const sel of ['.resizer.v[data-resize="left"]', '.resizer.v[data-resize="right"]', "#term-resizer"]) {
  const el = document.querySelector(sel);
  if (el) el.addEventListener("pointerup", saveLayout);
}
restoreLayout();

// -- main canvas -----------------------------------------------------------
const canvas = document.getElementById("canvas");
const canvasWrap = document.getElementById("canvas-wrap");
const ctx = canvas.getContext("2d");

// -- game board rendering --------------------------------------------------
// Board selection is driven by the left-panel Control Panel. Only snake and
// 2048 draw a real board; other environments fall back to the grid placeholder.
// (Illustrative/static for now — not yet wired to the engine.)
const boardState = { env: "2048", snake: { w: 20, h: 15 } };
let liveState = null;        // latest training render_state (snake/2048); overrides the static board
let replayTimer = null;      // interval id while animating a champion's replay

function drawCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // background
  ctx.fillStyle = "#0b0f16";
  ctx.fillRect(0, 0, w, h);

  if (liveState) {           // training is driving the board
    if (liveState.env === "snake") return drawSnakeLive(w, h, liveState);
    if (liveState.env === "2048") return draw2048Live(w, h, liveState);
  }
  if (boardState.env === "snake") return drawSnakeBoard(w, h);
  if (boardState.env === "2048") return draw2048Board(w, h);
  drawPlaceholder(w, h);
}

// Fallback for environments without a board yet (cartpole, humanoid, language).
function drawPlaceholder(w, h) {
  ctx.strokeStyle = "rgba(78,161,255,0.08)";
  ctx.lineWidth = 1;
  const step = 32;
  for (let x = (w % step) / 2; x < w; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = (h % step) / 2; y < h; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

  ctx.fillStyle = "rgba(215,221,229,0.35)";
  ctx.font = '600 15px "Cascadia Code", Consolas, monospace';
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("GENREG — no game board for this environment", w / 2, h / 2 - 10);
  ctx.font = '12px "Cascadia Code", Consolas, monospace';
  ctx.fillStyle = "rgba(125,135,148,0.6)";
  ctx.fillText(`${Math.round(w)} × ${Math.round(h)}`, w / 2, h / 2 + 14);
}

// A centered board area sized to fit cols × rows cells inside the canvas.
function boardRect(w, h, cols, rows, pad) {
  pad = pad == null ? 24 : pad;
  const cell = Math.max(4, Math.floor(Math.min((w - pad * 2) / cols, (h - pad * 2) / rows)));
  const bw = cell * cols, bh = cell * rows;
  return { x: Math.round((w - bw) / 2), y: Math.round((h - bh) / 2), cell, bw, bh };
}

function roundRectPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function boardCaption(w, r, text) {
  ctx.fillStyle = "rgba(125,135,148,0.7)";
  ctx.font = '12px "Cascadia Code", Consolas, monospace';
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, r.y + r.bh + 18);
}

// Shared snake renderer. `cells` is head-first [[x,y],...]; `food` is [x,y]|null.
function renderSnake(w, h, cols, rows, cells, food, caption) {
  const r = boardRect(w, h, cols, rows);

  ctx.fillStyle = "#0d1117";
  ctx.fillRect(r.x, r.y, r.bw, r.bh);

  ctx.strokeStyle = "rgba(78,161,255,0.10)";
  ctx.lineWidth = 1;
  for (let c = 0; c <= cols; c++) { const x = r.x + c * r.cell + 0.5; ctx.beginPath(); ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.bh); ctx.stroke(); }
  for (let rr = 0; rr <= rows; rr++) { const y = r.y + rr * r.cell + 0.5; ctx.beginPath(); ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.bw, y); ctx.stroke(); }

  const cell = (cx, cy, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(r.x + cx * r.cell + 1, r.y + cy * r.cell + 1, r.cell - 2, r.cell - 2);
  };
  if (food) cell(food[0], food[1], "#f85149");
  if (cells) {
    for (let i = cells.length - 1; i >= 0; i--) {
      const t = cells.length > 1 ? i / (cells.length - 1) : 0;   // tail(1) -> head(0)
      cell(cells[i][0], cells[i][1], i === 0 ? "#56d364" : `rgba(63,185,80,${0.55 + 0.45 * (1 - t)})`);
    }
  }

  ctx.strokeStyle = "rgba(78,161,255,0.45)";
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.bw, r.bh);
  boardCaption(w, r, caption);
}

function drawSnakeBoard(w, h) {          // static empty playfield
  renderSnake(w, h, boardState.snake.w, boardState.snake.h, null, null,
              `Snake — ${boardState.snake.w} × ${boardState.snake.h}`);
}

function drawSnakeLive(w, h, s) {        // live training frame
  const len = s.snake ? s.snake.length : 0;
  renderSnake(w, h, s.w, s.h, s.snake, s.food,
              `Snake — score ${s.score} · len ${len}${s.alive ? "" : " · dead"}`);
}

const TILE_COLORS = {
  0: "rgba(255,255,255,0.05)",
  2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", 32: "#f67c5f",
  64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", 512: "#edc850",
  1024: "#edc53f", 2048: "#edc22e",
};

// Shared 2048 renderer for a 4x4 `grid` of tile values.
function render2048(w, h, grid, caption) {
  const N = 4;
  const r = boardRect(w, h, N, N, 32);
  const gap = Math.max(5, Math.round(r.cell * 0.12));

  roundRectPath(r.x - gap, r.y - gap, r.bw + gap * 2, r.bh + gap * 2, 10);
  ctx.fillStyle = "#1c2330"; ctx.fill();

  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let ry = 0; ry < N; ry++) for (let cx = 0; cx < N; cx++) {
    const v = grid[ry][cx];
    const x = r.x + cx * r.cell + gap / 2, y = r.y + ry * r.cell + gap / 2, s = r.cell - gap;
    roundRectPath(x, y, s, s, 6);
    ctx.fillStyle = TILE_COLORS[v] || TILE_COLORS[2048];
    ctx.fill();
    if (v) {
      ctx.fillStyle = v <= 4 ? "#776e65" : "#f9f6f2";
      const digits = String(v).length;
      ctx.font = `700 ${Math.round(s * (digits > 3 ? 0.30 : 0.4))}px "Cascadia Code", Consolas, monospace`;
      ctx.fillText(String(v), x + s / 2, y + s / 2 + 1);
    }
  }
  boardCaption(w, r, caption);
}

function draw2048Board(w, h) {           // static illustrative position
  render2048(w, h, [[2, 0, 0, 4], [0, 4, 8, 0], [0, 0, 2, 0], [16, 0, 0, 2]], "2048 — 4 × 4");
}

function draw2048Live(w, h, s) {         // live training frame
  render2048(w, h, s.grid, `2048 — score ${s.score} · max ${s.max_tile}${s.over ? " · over" : ""}`);
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  drawCanvas();
}

new ResizeObserver(resizeCanvas).observe(canvasWrap);
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// -- control panel -> board wiring -----------------------------------------
// The Environment selector picks the board; the snake W/H inputs resize it.
// Only snake/2048 have a board; everything else shows the placeholder. Guard
// each lookup so a missing/stale template can't throw and blank the canvas.
(() => {
  const envSel = document.getElementById("cp-environment");
  const snakeW = document.getElementById("cp-snake-w");
  const snakeH = document.getElementById("cp-snake-h");
  const gc = document.getElementById("game-controls");
  if (!envSel) return;

  const clampInt = (val, lo, hi, dflt) => {
    const n = parseInt(val, 10);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
  };

  function showGamePanel(env) {
    if (!gc) return;
    const kind = env === "snake" ? "snake" : env === "2048" ? "2048" : "other";
    for (const p of gc.querySelectorAll(".gc-panel")) p.hidden = p.dataset.env !== kind;
  }

  function syncSnakeDims() {
    if (snakeW) boardState.snake.w = clampInt(snakeW.value, 5, 60, 20);
    if (snakeH) boardState.snake.h = clampInt(snakeH.value, 5, 60, 15);
  }

  function apply() {
    boardState.env = envSel.value;
    showGamePanel(boardState.env);
    syncSnakeDims();
    drawCanvas();
  }

  // manual env change drops any stale live/replay board so the static board shows
  envSel.addEventListener("change", () => { stopReplay(); liveState = null; apply(); });
  for (const el of [snakeW, snakeH]) {
    if (!el) continue;
    el.addEventListener("input", () => { syncSnakeDims(); if (boardState.env === "snake") drawCanvas(); });
  }

  apply();   // initialize board + game-controls panel from current selection
})();

// -- board API for the trainer (training.js) -------------------------------
function stopReplay() {
  if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
}

const boardAPI = {
  // push a single live frame (render_state dict)
  setLive(state) { stopReplay(); liveState = state; drawCanvas(); },
  // clear live mode; fall back to the static board
  clearLive() { stopReplay(); liveState = null; drawCanvas(); },
  // animate a champion's replay: {env, frames:[render_state,...]}
  playReplay(replay, fps) {
    stopReplay();
    const frames = (replay && replay.frames) || [];
    if (!frames.length) return;
    let i = 0;
    liveState = frames[0]; drawCanvas();
    replayTimer = setInterval(() => {
      i++;
      if (i >= frames.length) { liveState = frames[frames.length - 1]; drawCanvas(); stopReplay(); return; }
      liveState = frames[i];
      drawCanvas();
    }, 1000 / (fps || 15));
  },
  isReplaying() { return replayTimer != null; },
};

// expose for other scripts / console tinkering
window.GENREG = { canvas, ctx, drawCanvas, resizeCanvas, boardState, board: boardAPI };
