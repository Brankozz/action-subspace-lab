/* A small, dependency-free world-coordinate scene viewer for the Action Subspace Atlas.
 * Objects are simplified static geometry; paths are supplied by the caller.
 * No physics, inverse kinematics, success inference, or unit conversion happens here.
 */
(function (root) {
  'use strict';

  const TAU = Math.PI * 2;
  const C = { paper: '#fffefa', ink: '#23372f', muted: '#78867e', grid: '#dce5dc', manual: '#167866', candidate: '#8a6daf' };
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const point = v => v && v.length >= 3 && [v[0], v[1], v[2]].every(finite);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const norm = a => Math.hypot(a[0], a[1], a[2]);
  const safeColor = (value, fallback) => typeof value === 'string' && /^#[\da-f]{3}(?:[\da-f]{3})?$/i.test(value) ? value : fallback;
  function tint(hex, factor) {
    hex = hex.slice(1); if (hex.length === 3) hex = hex.split('').map(x => x + x).join('');
    return '#' + [0, 2, 4].map(i => Math.round(clamp(parseInt(hex.slice(i, i + 2), 16) * factor, 0, 255)).toString(16).padStart(2, '0')).join('');
  }
  function roleColor(o) {
    return safeColor(o.color, /goal|destination|region/.test(o.role || '') ? '#cf9a45' : /target|object_of_interest|manipuland/.test(o.role || '') ? '#498f83' : /table|floor/.test(o.role || '') ? '#c8c5ae' : /fixture|cabinet|drawer|basket/.test(o.role || '') ? '#bc9e75' : '#a9b4ae');
  }
  function transform(v, o) {
    const r = o.rotation, c = o.center;
    return r && r.length === 9 && Array.from(r).every(finite)
      ? [c[0] + r[0] * v[0] + r[1] * v[1] + r[2] * v[2], c[1] + r[3] * v[0] + r[4] * v[1] + r[5] * v[2], c[2] + r[6] * v[0] + r[7] * v[1] + r[8] * v[2]]
      : add(c, v);
  }
  function geometry(o) {
    if (!o || !point(o.center) || !point(o.size) || o.size.some(v => v <= 0) || o.visible === false) return null;
    const half = o.size.map(v => v / 2), vertices = [], faces = [];
    if (o.shape === 'cylinder') {
      const n = 24;
      for (const z of [-half[2], half[2]]) for (let i = 0; i < n; i++) vertices.push([half[0] * Math.cos(TAU * i / n), half[1] * Math.sin(TAU * i / n), z]);
      faces.push(Array.from({ length: n }, (_, i) => n - 1 - i), Array.from({ length: n }, (_, i) => i + n));
      for (let i = 0; i < n; i++) faces.push([i, (i + 1) % n, (i + 1) % n + n, i + n]);
    } else if (o.shape === 'sphere') {
      const n = 20, m = 10;
      for (let j = 0; j <= m; j++) for (let i = 0; i < n; i++) {
        const a = TAU * i / n, b = Math.PI * j / m;
        vertices.push([half[0] * Math.sin(b) * Math.cos(a), half[1] * Math.sin(b) * Math.sin(a), half[2] * Math.cos(b)]);
      }
      for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) faces.push([j * n + i, j * n + (i + 1) % n, (j + 1) * n + (i + 1) % n, (j + 1) * n + i]);
    } else {
      for (const z of [-half[2], half[2]]) for (const y of [-half[1], half[1]]) for (const x of [-half[0], half[0]]) vertices.push([x, y, z]);
      faces.push([0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]);
    }
    const worldVertices = vertices.map(v => transform(v, o)), color = roleColor(o);
    const paintFaces = faces.map(indices => {
      const [p, q, r] = indices.slice(0, 3).map(i => worldVertices[i]);
      const u = q.map((v, d) => v - p[d]), v = r.map((value, d) => value - p[d]);
      const normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]], length = norm(normal) || 1;
      const lighting = .86 + .15 * Math.abs(normal[2] / length) + .06 * Math.abs(normal[0] / length), fill = tint(color, lighting);
      return { indices, fill, edge: tint(fill, .71), round: o.shape === 'sphere' || (o.shape === 'cylinder' && indices.length === 4) };
    });
    return { object: o, vertices: worldVertices, paintFaces, color };
  }
  function normalizedBounds(value) {
    if (!value) return null;
    const lo = value.min || value[0], hi = value.max || value[1];
    if (!point(lo) || !point(hi)) return null;
    return { min: [0, 1, 2].map(i => Math.min(lo[i], hi[i])), max: [0, 1, 2].map(i => Math.max(lo[i], hi[i])) };
  }
  function distanceToSegment(x, y, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], t = clamp(((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy || 1), 0, 1);
    return Math.hypot(x - a[0] - dx * t, y - a[1] - dy * t);
  }
  function niceStep(value) {
    const power = Math.pow(10, Math.floor(Math.log10(Math.max(value, 1e-6)))), v = value / power;
    return power * (v < 1.5 ? 1 : v < 3.5 ? 2 : v < 7.5 ? 5 : 10);
  }
  function tickLabel(v, step) {
    const digits = clamp(-Math.floor(Math.log10(step)), 0, 3);
    return (Math.abs(v) < step * .01 ? 0 : v).toFixed(digits);
  }

  function create(canvas, callbacks = {}) {
    if (!canvas || typeof canvas.getContext !== 'function') throw Error('AtlasSceneView requires a canvas.');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw Error('A 2D canvas context is not available.');
    const state = { yaw: -.66, pitch: .64, zoom: 1, pan: [0, 0], view: 'perspective', progress: 1 };
    let scene = null, paths = [], objects = [], options = {}, frame = null, frameKey = null;
    let width = 1, height = 1, scale = 1, dpr = 1, dead = false, scheduled = 0, hits = [], hover = null;
    let drag = null, pinch = null;
    const pointers = new Map(), disposers = [];
    const previous = { cursor: canvas.style.cursor, touchAction: canvas.style.touchAction, tabIndex: canvas.getAttribute('tabindex'), ariaLabel: canvas.getAttribute('aria-label') };
    canvas.style.cursor = 'grab'; canvas.style.touchAction = 'none';
    if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
    if (!canvas.getAttribute('aria-label')) canvas.setAttribute('aria-label', '任务的三维几何场景与末端轨迹。拖动旋转，滚轮缩放，Shift 或右键拖动平移。方向键旋转，Home 复位。');

    function on(name, fn, config) { canvas.addEventListener(name, fn, config); disposers.push(() => canvas.removeEventListener(name, fn, config)); }
    function invalidate() {
      if (!scheduled && !dead) scheduled = root.requestAnimationFrame(() => { scheduled = 0; draw(); });
    }
    function setFrame(key) {
      let bounds = normalizedBounds(options.frameBounds || scene?.bounds);
      if (!bounds) {
        let pts = objects.flatMap(g => g.vertices);
        // A sampled outlier must not change the camera framing or hide scene detail.
        const base = paths.filter(p => p.kind !== 'candidate').flatMap(p => p.points || []).filter(point);
        pts = pts.concat(base.length ? base : paths.flatMap(p => p.points || []).filter(point));
        if (!pts.length) pts = [[-.35, -.35, 0], [.35, .35, .7]];
        bounds = { min: [0, 1, 2].map(d => Math.min(...pts.map(p => p[d]))), max: [0, 1, 2].map(d => Math.max(...pts.map(p => p[d]))) };
      }
      const center = bounds.min.map((v, d) => (v + bounds.max[d]) / 2);
      const radius = Math.max(.04, norm(bounds.max.map((v, d) => (v - bounds.min[d]) / 2)));
      frame = { bounds, center, radius }; frameKey = key;
    }
    function project(v) {
      const x = v[0] - frame.center[0], y = v[1] - frame.center[1], z = v[2] - frame.center[2];
      const right = x * Math.cos(state.yaw) - y * Math.sin(state.yaw), forward = x * Math.sin(state.yaw) + y * Math.cos(state.yaw);
      return [width / 2 + right * scale + state.pan[0], height / 2 + 4 - (z * Math.cos(state.pitch) - forward * Math.sin(state.pitch)) * scale + state.pan[1], forward * Math.cos(state.pitch) + z * Math.sin(state.pitch)];
    }
    function stroke(ps, color, lineWidth = 1, opacity = 1, dash = []) {
      if (!ps.length || opacity <= 0) return;
      ctx.save(); ctx.globalAlpha = opacity; ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.setLineDash(dash);
      ctx.beginPath(); ps.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke(); ctx.restore();
    }
    function dot(p, r, color, hollow = false, alpha = 1) {
      ctx.save(); ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, TAU);
      ctx.fillStyle = hollow ? C.paper : color; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke(); ctx.restore();
    }
    function roundBox(x, y, w, h, radius = 5) {
      ctx.beginPath(); ctx.moveTo(x + radius, y); ctx.arcTo(x + w, y, x + w, y + h, radius); ctx.arcTo(x + w, y + h, x, y + h, radius); ctx.arcTo(x, y + h, x, y, radius); ctx.arcTo(x, y, x + w, y, radius); ctx.closePath();
    }
    function drawGrid() {
      if (options.showGrid === false) return;
      const bounds = frame.bounds, table = objects.find(g => g.object.role === 'table');
      const z = finite(scene?.tableZ) ? scene.tableZ : table ? Math.max(...table.vertices.map(p => p[2])) + .0002 : bounds.min[2];
      const span = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], .12), step = niceStep(span / 6);
      const xmin = Math.floor(bounds.min[0] / step) * step, xmax = Math.ceil(bounds.max[0] / step) * step;
      const ymin = Math.floor(bounds.min[1] / step) * step, ymax = Math.ceil(bounds.max[1] / step) * step;
      ctx.save(); ctx.font = '9px ui-monospace, SFMono-Regular, monospace'; ctx.fillStyle = C.muted;
      for (let x = xmin, n = 0; x <= xmax + step / 100 && n < 30; x += step, n++) {
        const a = project([x, ymin, z]), b = project([x, ymax, z]); stroke([a, b], C.grid, .75, .75);
        if (n % 2 === 0 && state.zoom < 2) { ctx.textAlign = 'center'; ctx.fillText(tickLabel(x, step), a[0], a[1] + 13); }
      }
      for (let y = ymin, n = 0; y <= ymax + step / 100 && n < 30; y += step, n++) {
        const a = project([xmin, y, z]), b = project([xmax, y, z]); stroke([a, b], C.grid, .75, .75);
        if (n % 2 === 0 && state.zoom < 2) { ctx.textAlign = 'right'; ctx.fillText(tickLabel(y, step), a[0] - 5, a[1] + 3); }
      }
      const corner = project([xmin, ymax, z]), topZ = Math.max(bounds.max[2], z + .1);
      stroke([corner, project([xmin, ymax, topZ])], '#c5d3c8', 1, .8, [2, 4]);
      if (Math.cos(state.pitch) > .15) {
        const p = project([xmin, ymax, topZ]); ctx.textAlign = 'right'; ctx.fillText('z ' + tickLabel(topZ, step), p[0] - 5, p[1] - 5);
      }
      ctx.restore();
    }
    function drawObjects() {
      if (options.showObjects === false) return;
      const faces = [], opacity = clamp(finite(options.objectOpacity) ? options.objectOpacity : .28, 0, 1);
      for (const g of objects) {
        const projected = g.vertices.map(project), isTable = /table|floor/.test(g.object.role || ''), a = opacity * (isTable ? .4 : 1);
        for (const face of g.paintFaces) {
          const ps = face.indices.map(i => projected[i]);
          faces.push({ ps, depth: ps.reduce((sum, p) => sum + p[2], 0) / ps.length, color: face.fill, edge: face.edge, alpha: a, edgeAlpha: isTable ? opacity * .7 : opacity * 1.4, round: face.round });
        }
      }
      faces.sort((a, b) => a.depth - b.depth);
      for (const f of faces) {
        ctx.save(); ctx.globalAlpha = f.alpha; ctx.fillStyle = f.color; ctx.beginPath(); f.ps.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill();
        ctx.globalAlpha = clamp(f.edgeAlpha, 0, .8) * (f.round ? .2 : 1); ctx.strokeStyle = f.edge; ctx.lineWidth = .8; ctx.stroke(); ctx.restore();
      }
    }
    function clippedPoints(ps) {
      if (ps.length < 2) return ps;
      const at = state.progress * (ps.length - 1), index = Math.floor(at), out = ps.slice(0, index + 1);
      if (index + 1 < ps.length && at > index) out.push(ps[index].map((v, d) => v + (ps[index + 1][d] - v) * (at - index)));
      return out;
    }
    function drawPaths() {
      hits = [];
      const priority = p => p.selected ? 5 : p.kind === 'manual' ? 4 : p.kind === 'mean' ? 3 : p.kind === 'reference' ? 2 : 1;
      const visible = paths.filter(p => !(p.kind === 'candidate' && options.showAllCandidates === false && !p.selected)).sort((a, b) => priority(a) - priority(b));
      const globalOpacity = clamp(finite(options.trajectoryOpacity) ? options.trajectoryOpacity : 1, 0, 1), globalWidth = finite(options.lineWidth) ? clamp(options.lineWidth, .3, 8) : 1;
      for (const path of visible) {
        const source = (path.points || []).filter(point), ps = clippedPoints(source).map(project);
        if (!ps.length) continue;
        const color = safeColor(path.color, path.kind === 'candidate' ? C.candidate : path.kind === 'manual' ? C.manual : C.ink);
        const chosen = !!path.selected, faded = path.kind === 'candidate' && !chosen;
        const alpha = clamp(finite(path.opacity) ? path.opacity : faded ? .16 : 1, 0, 1) * globalOpacity;
        const w = (finite(path.width) ? path.width : faded ? 1.1 : chosen ? 2.8 : 2) * globalWidth;
        if (!faded) stroke(ps, C.paper, w + 2.3, alpha * .72);
        stroke(ps, color, w, alpha, path.kind === 'mean' ? [5, 4] : []);
        if (path.kind === 'candidate') hits.push({ path, points: ps });
        if (!faded) {
          dot(ps[0], chosen ? 3.9 : 2.9, color, true, alpha);
          const end = ps[ps.length - 1]; dot(end, chosen ? 4.1 : 3, color, false, alpha);
          if (chosen && state.progress < 1) {
            ctx.save(); ctx.globalAlpha = alpha * .25; ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(end[0], end[1], 8, 0, TAU); ctx.stroke(); ctx.restore();
          }
        }
      }
    }
    function drawInitialEef() {
      if (options.showInitialEef === false || !point(scene?.eefStart)) return;
      const p = project(scene.eefStart);
      if (p[0] < -12 || p[0] > width + 12 || p[1] < 0 || p[1] > height) return;
      ctx.save(); ctx.strokeStyle = C.ink; ctx.fillStyle = C.paper; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(p[0], p[1] - 5); ctx.lineTo(p[0] + 5, p[1]); ctx.lineTo(p[0], p[1] + 5); ctx.lineTo(p[0] - 5, p[1]); ctx.closePath(); ctx.fill(); ctx.stroke();
      stroke([[p[0] - 8, p[1]], [p[0] + 8, p[1]]], C.ink, .8, .7);
      stroke([[p[0], p[1] - 8], [p[0], p[1] + 8]], C.ink, .8, .7);
      if (options.showLabels !== false) {
        ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        const evaluation = /evaluation/.test(scene.provenance?.kind || '');
        const label = evaluation ? '评测初始 EEF' : '参考布局初始 EEF', w = ctx.measureText(label).width + 12;
        const x = clamp(p[0] + 12, 8, width - w - 8), y = clamp(p[1] - 24, 35, height - 48);
        roundBox(x, y, w, 20); ctx.fillStyle = 'rgba(255,254,250,.92)'; ctx.fill(); ctx.fillStyle = C.ink; ctx.fillText(label, x + 6, y + 10);
      }
      ctx.restore();
    }
    function fitText(text, maxWidth) {
      let value = String(text || ''); if (ctx.measureText(value).width <= maxWidth) return value;
      while (value.length && ctx.measureText(value + '…').width > maxWidth) value = value.slice(0, -1);
      return value + '…';
    }
    function drawLabels() {
      if (options.showLabels === false || options.showObjects === false) return;
      const rects = [], labels = objects.filter(g => g.object.label && !/table|floor/.test(g.object.role || ''));
      labels.sort((a, b) => Number(/target|goal|destination/.test(b.object.role || '')) - Number(/target|goal|destination/.test(a.object.role || '')));
      ctx.save(); ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      for (const g of labels.slice(0, width < 420 ? 6 : 10)) {
        const top = g.vertices.reduce((a, b) => a[2] > b[2] ? a : b), anchor = project(point(g.object.labelPosition) ? g.object.labelPosition : [g.object.center[0], g.object.center[1], top[2]]);
        if (anchor[0] < -30 || anchor[0] > width + 30 || anchor[1] < -30 || anchor[1] > height + 30) continue;
        const text = fitText(g.object.label, Math.min(136, width * .34)), w = ctx.measureText(text).width + 14, h = 21;
        let placement = null;
        for (const [dx, dy] of [[10, -27], [10, 6], [-w - 10, -27], [-w - 10, 6], [10, -51], [-w - 10, -51]]) {
          const x = clamp(anchor[0] + dx, 8, width - w - 8), y = clamp(anchor[1] + dy, 37, height - h - 48);
          const r = { x, y, w, h };
          if (!rects.some(q => r.x < q.x + q.w + 3 && r.x + r.w + 3 > q.x && r.y < q.y + q.h + 3 && r.y + r.h + 3 > q.y)) { placement = r; break; }
        }
        if (!placement) continue;
        const { x, y } = placement; rects.push(placement);
        stroke([anchor, [clamp(anchor[0], x, x + w), y + h / 2]], g.color, .8, .65);
        roundBox(x, y, w, h); ctx.fillStyle = 'rgba(255,254,250,.92)'; ctx.fill(); ctx.strokeStyle = tint(g.color, .94); ctx.lineWidth = .6; ctx.stroke();
        ctx.fillStyle = tint(g.color, .63); ctx.fillText(text, x + 7, y + h / 2);
      }
      ctx.restore();
    }
    function drawOverlay() {
      ctx.save(); ctx.font = '10px system-ui, sans-serif'; ctx.fillStyle = C.muted; ctx.textAlign = 'left';
      ctx.fillText(options.coordinateLabel || '世界坐标 · m', 14, 20);
      ctx.textAlign = 'right'; ctx.fillText(state.view === 'top' ? '俯视 · 等比例' : state.view === 'front' ? '正视 · 等比例' : '三维 · 等比例', width - 14, 20);
      // A screen-anchored orientation triad remains visible when the world origin is off screen.
      const origin = [38, height - 49], center = project(frame.center), len = 22 / scale;
      const axes = [[len, 0, 0], [0, len, 0], [0, 0, len]], colors = ['#a66c57', '#7a9466', '#628696'];
      ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
      axes.forEach((v, i) => { const q = project(add(frame.center, v)), end = [origin[0] + q[0] - center[0], origin[1] + q[1] - center[1]]; stroke([origin, end], colors[i], 1.5); dot(end, 1.4, colors[i]); ctx.fillStyle = colors[i]; ctx.fillText('xyz'[i], end[0] + (end[0] >= origin[0] ? 7 : -7), end[1] - 3); });
      ctx.font = '9px system-ui, sans-serif'; ctx.fillStyle = C.muted; ctx.textAlign = 'right';
      const help = width < 440 ? '拖动旋转 · 双指缩放 / 平移' : '拖动旋转 · 滚轮缩放 · Shift / 右键平移'; ctx.fillText(help, width - 14, height - 14);
      if (state.progress < 1) { ctx.textAlign = 'right'; ctx.fillStyle = C.ink; ctx.fillText('轨迹进度 ' + Math.round(state.progress * 100) + '%', width - 14, height - 31); }
      if (!objects.length && !paths.some(p => p.points?.length)) { ctx.textAlign = 'center'; ctx.font = '12px system-ui, sans-serif'; ctx.fillText('等待场景与轨迹', width / 2, height / 2); }
      if (hover) {
        ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left';
        const label = fitText(hover.path.label || '采样候选 ' + hover.path.id, width - 48), w = ctx.measureText(label).width + 16;
        const x = clamp(hover.x + 13, 8, width - w - 8), y = clamp(hover.y - 31, 31, height - 35);
        roundBox(x, y, w, 25); ctx.fillStyle = 'rgba(35,55,47,.93)'; ctx.fill(); ctx.fillStyle = '#fffefa'; ctx.fillText(label, x + 8, y + 17);
      }
      ctx.restore();
    }
    function draw() {
      if (dead || !frame) return;
      const rect = canvas.getBoundingClientRect(); width = Math.max(1, rect.width); height = Math.max(1, rect.height); dpr = Math.min(root.devicePixelRatio || 1, 2);
      const pw = Math.round(width * dpr), ph = Math.round(height * dpr);
      if (canvas.width !== pw) canvas.width = pw; if (canvas.height !== ph) canvas.height = ph;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height); ctx.fillStyle = C.paper; ctx.fillRect(0, 0, width, height);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      scale = Math.max(1, Math.min(width - 64, height - 94)) / (frame.radius * 2.15) * state.zoom;
      drawGrid(); drawObjects(); drawPaths(); drawLabels(); drawInitialEef(); drawOverlay();
    }
    function pick(x, y) {
      let result = null, distance = 9;
      for (const hit of hits) for (let i = 1; i < hit.points.length; i++) { const d = distanceToSegment(x, y, hit.points[i - 1], hit.points[i]); if (d < distance) { distance = d; result = hit.path; } }
      return result;
    }
    function local(event) { const b = canvas.getBoundingClientRect(); return [event.clientX - b.left, event.clientY - b.top]; }
    function touchPair() {
      const [a, b] = [...pointers.values()];
      return a && b ? { distance: Math.max(1, Math.hypot(a[0] - b[0], a[1] - b[1])), center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] } : null;
    }
    on('pointerdown', e => {
      if (e.button > 2) return; const p = local(e); pointers.set(e.pointerId, p);
      canvas.setPointerCapture(e.pointerId); canvas.focus({ preventScroll: true });
      drag = { id: e.pointerId, start: p, last: p, moved: false, pan: e.button === 2 || e.button === 1 || e.shiftKey };
      pinch = touchPair(); if (pinch) drag.moved = true;
      hover = null; canvas.style.cursor = 'grabbing'; e.preventDefault(); invalidate();
    });
    on('pointermove', e => {
      const p = local(e);
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
      if (drag) {
        const nextPinch = touchPair();
        if (pinch && nextPinch) { state.zoom = clamp(state.zoom * nextPinch.distance / pinch.distance, .3, 12); state.pan[0] += nextPinch.center[0] - pinch.center[0]; state.pan[1] += nextPinch.center[1] - pinch.center[1]; pinch = nextPinch; drag.moved = true; }
        else if (e.pointerId === drag.id) {
          const dx = p[0] - drag.last[0], dy = p[1] - drag.last[1];
          if (Math.hypot(p[0] - drag.start[0], p[1] - drag.start[1]) > 3) drag.moved = true;
          if (drag.pan || e.shiftKey) { state.pan[0] += dx; state.pan[1] += dy; }
          else { state.yaw += dx * .008; state.pitch = clamp(state.pitch + dy * .008, -.45, Math.PI / 2); state.view = 'perspective'; }
          drag.last = p;
        }
        invalidate(); return;
      }
      const path = pick(p[0], p[1]); hover = path ? { path, x: p[0], y: p[1] } : null; canvas.style.cursor = path ? 'pointer' : 'grab'; invalidate();
    });
    function finish(e, cancelled = false) {
      const p = local(e), wasDrag = drag;
      pointers.delete(e.pointerId); if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      if (!cancelled && wasDrag && !wasDrag.moved && !wasDrag.pan && !pinch) { const selected = pick(p[0], p[1]); if (selected && callbacks.onSelectCandidate) callbacks.onSelectCandidate(selected.id, selected); }
      drag = null; pinch = null;
      if (pointers.size) { const [id, last] = [...pointers.entries()][0]; drag = { id, start: last, last, moved: true, pan: false }; }
      canvas.style.cursor = drag ? 'grabbing' : 'grab'; invalidate();
    }
    on('pointerup', e => finish(e)); on('pointercancel', e => finish(e, true));
    on('pointerleave', () => { if (!drag) { hover = null; invalidate(); } });
    on('contextmenu', e => e.preventDefault());
    on('wheel', e => { e.preventDefault(); const factor = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1; state.zoom = clamp(state.zoom * Math.exp(-e.deltaY * factor * .0012), .3, 12); hover = null; invalidate(); }, { passive: false });
    on('dblclick', e => { e.preventDefault(); reset(); });
    on('keydown', e => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === 'Home' || e.key === '0') reset();
      else if (e.key === '+' || e.key === '=') state.zoom = clamp(state.zoom * 1.15, .3, 12);
      else if (e.key === '-') state.zoom = clamp(state.zoom / 1.15, .3, 12);
      else if (e.key === '[' || e.key === ']') { state.progress = clamp(state.progress + (e.key === ']' ? .02 : -.02), 0, 1); if (callbacks.onProgress) callbacks.onProgress(state.progress); }
      else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
        if (e.shiftKey) { state.pan[0] += e.key === 'ArrowLeft' ? -12 : e.key === 'ArrowRight' ? 12 : 0; state.pan[1] += e.key === 'ArrowUp' ? -12 : e.key === 'ArrowDown' ? 12 : 0; }
        else { state.yaw += e.key === 'ArrowLeft' ? -.1 : e.key === 'ArrowRight' ? .1 : 0; state.pitch = clamp(state.pitch + (e.key === 'ArrowUp' ? .1 : e.key === 'ArrowDown' ? -.1 : 0), -.45, Math.PI / 2); state.view = 'perspective'; }
      } else return;
      e.preventDefault(); invalidate();
    });
    const observer = typeof root.ResizeObserver === 'function' ? new root.ResizeObserver(invalidate) : null;
    if (observer) observer.observe(canvas); else { root.addEventListener('resize', invalidate); disposers.push(() => root.removeEventListener('resize', invalidate)); }
    function update(data = {}) {
      if (dead) return;
      if ('scene' in data) { scene = data.scene; objects = (scene?.objects || []).map(geometry).filter(Boolean); }
      if ('paths' in data) paths = Array.isArray(data.paths) ? data.paths : [];
      if ('options' in data) options = { ...options, ...data.options };
      if (finite(data.progress)) state.progress = clamp(data.progress, 0, 1);
      const bounds = options.frameBounds || scene?.bounds;
      const key = String(options.frameKey ?? scene?.key ?? scene?.id ?? '') + '|' + JSON.stringify(bounds || objects.map(g => [g.object.id, g.object.center, g.object.size]));
      if (!frame || key !== frameKey) { setFrame(key); state.pan = [0, 0]; state.zoom = 1; }
      hover = null; draw();
    }
    function reset() { state.yaw = -.66; state.pitch = .64; state.zoom = 1; state.pan = [0, 0]; state.view = 'perspective'; hover = null; invalidate(); }
    function setView(view) {
      if (!['perspective', 'top', 'front'].includes(view)) return;
      state.view = view; state.yaw = view === 'top' || view === 'front' ? 0 : -.66; state.pitch = view === 'top' ? Math.PI / 2 : view === 'front' ? 0 : .64;
      state.pan = [0, 0]; state.zoom = 1; hover = null; invalidate();
    }
    function getState() { return { ...state, pan: state.pan.slice(), frame: frame ? JSON.parse(JSON.stringify(frame)) : null, width, height, objectCount: objects.length, pathCount: paths.length }; }
    function destroy() {
      dead = true; if (scheduled) root.cancelAnimationFrame(scheduled); observer?.disconnect(); disposers.forEach(fn => fn());
      canvas.style.cursor = previous.cursor; canvas.style.touchAction = previous.touchAction;
      for (const [key, value] of [['tabindex', previous.tabIndex], ['aria-label', previous.ariaLabel]]) if (value === null) canvas.removeAttribute(key); else canvas.setAttribute(key, value);
      paths = []; objects = []; pointers.clear(); hits = []; hover = null;
    }
    return { update, reset, setView, getState, destroy };
  }
  root.AtlasSceneView = Object.freeze({ create });
})(typeof window !== 'undefined' ? window : globalThis);
