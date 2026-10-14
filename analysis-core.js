/* Action Lab numeric core. Pure functions, no DOM: loaded by index.html and testable from Node.
 * A trajectory is a Float64Array of length L*D (time-major). A matrix is an array of Float64Array rows. */
(function (root) {
  'use strict';

  function makeRng(seed) {
    let a = seed >>> 0;
    const f = function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    f.normal = function () {
      let u = 0;
      while (u === 0) u = f();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * f());
    };
    f.permutation = function (n) {
      const p = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(f() * (i + 1));
        const tmp = p[i]; p[i] = p[j]; p[j] = tmp;
      }
      return p;
    };
    return f;
  }

  function resample(x, T, D, L) {
    const out = new Float64Array(L * D);
    for (let l = 0; l < L; l++) {
      const pos = T > 1 && L > 1 ? (l * (T - 1)) / (L - 1) : 0;
      const i0 = Math.min(Math.floor(pos), T - 1);
      const i1 = Math.min(i0 + 1, T - 1);
      const w = pos - i0;
      for (let d = 0; d < D; d++) out[l * D + d] = (1 - w) * x[i0 * D + d] + w * x[i1 * D + d];
    }
    return out;
  }

  /* comps: [{field, dims:[...], cumsum}] -> {T, D, data} in the episode's own time. */
  function buildSignal(ep, comps, fieldDims) {
    const T = ep.T;
    let D = 0;
    for (const c of comps) D += c.dims.length;
    const data = new Float64Array(T * D);
    let col = 0;
    for (const c of comps) {
      const src = ep.data[c.field];
      const fd = fieldDims[c.field];
      for (const dim of c.dims) {
        let acc = 0;
        for (let t = 0; t < T; t++) {
          const v = src[t * fd + dim];
          acc += v;
          data[t * D + col] = c.cumsum ? acc : v;
        }
        col++;
      }
    }
    return { T: T, D: D, data: data };
  }

  /* Symmetric eigendecomposition by cyclic Jacobi. A is n*n row-major and is overwritten. */
  function jacobi(A, n) {
    const V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    let scale = 0;
    for (let i = 0; i < n * n; i++) scale += A[i] * A[i];
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
      if (off <= 1e-22 * scale) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = A[p * n + q];
          if (Math.abs(apq) < 1e-300) continue;
          const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1);
          const s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = A[k * n + p], akq = A[k * n + q];
            A[k * n + p] = c * akp - s * akq;
            A[k * n + q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = A[p * n + k], aqk = A[q * n + k];
            A[p * n + k] = c * apk - s * aqk;
            A[q * n + k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k * n + p], vkq = V[k * n + q];
            V[k * n + p] = c * vkp - s * vkq;
            V[k * n + q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[j * n + j] - A[i * n + i]);
    const values = new Float64Array(n);
    const vectors = [];
    order.forEach((idx, r) => {
      values[r] = A[idx * n + idx];
      const v = new Float64Array(n);
      for (let k = 0; k < n; k++) v[k] = V[k * n + idx];
      vectors.push(v);
    });
    return { values: values, vectors: vectors };
  }

  /* Top-k eigenpairs of a PSD matrix: exact Jacobi when small, subspace iteration otherwise. */
  function eigTop(A, n, k) {
    if (n <= 220) {
      const full = jacobi(Float64Array.from(A), n);
      return { values: full.values, vectors: full.vectors, exact: true };
    }
    k = Math.min(k, n);
    const rnd = makeRng(12345);
    let Q = [];
    for (let j = 0; j < k; j++) {
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = rnd.normal();
      Q.push(v);
    }
    const orthonormalize = function (cols) {
      for (let j = 0; j < cols.length; j++) {
        for (let pass = 0; pass < 2; pass++) {
          for (let i = 0; i < j; i++) {
            let dot = 0;
            for (let r = 0; r < n; r++) dot += cols[i][r] * cols[j][r];
            for (let r = 0; r < n; r++) cols[j][r] -= dot * cols[i][r];
          }
        }
        let norm = 0;
        for (let r = 0; r < n; r++) norm += cols[j][r] * cols[j][r];
        norm = Math.sqrt(norm) || 1;
        for (let r = 0; r < n; r++) cols[j][r] /= norm;
      }
    };
    const multiply = function (cols) {
      return cols.map(function (q) {
        const z = new Float64Array(n);
        for (let r = 0; r < n; r++) {
          let s = 0;
          const row = r * n;
          for (let c = 0; c < n; c++) s += A[row + c] * q[c];
          z[r] = s;
        }
        return z;
      });
    };
    orthonormalize(Q);
    for (let it = 0; it < 24; it++) {
      Q = multiply(Q);
      orthonormalize(Q);
    }
    const Z = multiply(Q);
    const B = new Float64Array(k * k);
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += Q[i][r] * Z[j][r];
      B[i * k + j] = s;
    }
    const small = jacobi(B, k);
    const vectors = small.vectors.map(function (w) {
      const v = new Float64Array(n);
      for (let j = 0; j < k; j++) for (let r = 0; r < n; r++) v[r] += w[j] * Q[j][r];
      return v;
    });
    return { values: small.values, vectors: vectors, exact: false };
  }

  function meanOf(rows) {
    const M = rows[0].length;
    const mu = new Float64Array(M);
    for (const r of rows) for (let j = 0; j < M; j++) mu[j] += r[j];
    for (let j = 0; j < M; j++) mu[j] /= rows.length;
    return mu;
  }

  /* PCA of the deviations from the mean. Returns the top `kmax` unit components. */
  function pca(rows, kmax) {
    const N = rows.length, M = rows[0].length;
    const mean = meanOf(rows);
    const C = rows.map(function (r) {
      const c = new Float64Array(M);
      for (let j = 0; j < M; j++) c[j] = r[j] - mean[j];
      return c;
    });
    let total = 0;
    for (const c of C) for (let j = 0; j < M; j++) total += c[j] * c[j];
    let evals, comps = [], frob2 = 0, exact;
    if (N <= M) {
      const G = new Float64Array(N * N);
      for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
        let s = 0;
        const a = C[i], b = C[j];
        for (let m = 0; m < M; m++) s += a[m] * b[m];
        G[i * N + j] = s; G[j * N + i] = s;
      }
      for (let i = 0; i < N * N; i++) frob2 += G[i] * G[i];
      const e = eigTop(G, N, Math.max(kmax, 40));
      evals = e.values; exact = e.exact;
      for (let k = 0; k < Math.min(kmax, e.vectors.length); k++) {
        const lam = evals[k];
        const v = new Float64Array(M);
        if (lam > 1e-12 * Math.max(total, 1e-300)) {
          const inv = 1 / Math.sqrt(lam);
          for (let i = 0; i < N; i++) {
            const u = e.vectors[k][i] * inv, c = C[i];
            for (let m = 0; m < M; m++) v[m] += u * c[m];
          }
        }
        comps.push(v);
      }
    } else {
      const S = new Float64Array(M * M);
      for (const c of C) for (let i = 0; i < M; i++) {
        const ci = c[i];
        if (ci === 0) continue;
        const row = i * M;
        for (let j = i; j < M; j++) S[row + j] += ci * c[j];
      }
      for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) S[j * M + i] = S[i * M + j];
      for (let i = 0; i < M * M; i++) frob2 += S[i] * S[i];
      const e = eigTop(S, M, Math.max(kmax, 40));
      evals = e.values; exact = e.exact;
      comps = e.vectors.slice(0, kmax);
    }
    const vals = Float64Array.from(evals, function (v) { return Math.max(v, 0); });
    return { mean: mean, evals: vals, comps: comps, total: total, frob2: frob2, n: N, exact: exact };
  }

  function evrOf(model) {
    return Float64Array.from(model.evals, function (v) { return v / Math.max(model.total, 1e-300); });
  }

  /* Smallest k whose cumulative explained variance reaches `level`; null when not reached. */
  function kAt(evr, level) {
    let acc = 0;
    for (let k = 0; k < evr.length; k++) {
      acc += evr[k];
      if (acc >= level - 1e-9) return k + 1;
    }
    return null;
  }

  function participationRatio(model) {
    return (model.total * model.total) / Math.max(model.frob2, 1e-300);
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  /* Relative held-out residual for k = 0..kmax (1.0 at k=0 means "mean only"). */
  function heldoutCurve(rows, kmax, seed, folds) {
    folds = folds || 5;
    const N = rows.length;
    const order = makeRng(seed).permutation(N);
    const num = new Float64Array(kmax + 1);
    let den = 0;
    for (let f = 0; f < folds; f++) {
      const test = [], train = [];
      order.forEach(function (idx, pos) { (pos % folds === f ? test : train).push(rows[idx]); });
      if (!test.length || train.length < 2) continue;
      const model = pca(train, kmax);
      for (const x of test) {
        const r = new Float64Array(x.length);
        for (let j = 0; j < x.length; j++) r[j] = x[j] - model.mean[j];
        let left = dot(r, r);
        den += left;
        num[0] += left;
        for (let k = 1; k <= kmax; k++) {
          if (k - 1 < model.comps.length) {
            const c = dot(r, model.comps[k - 1]);
            left -= c * c;
          }
          num[k] += Math.max(left, 0);
        }
      }
    }
    return Float64Array.from(num, function (v) { return v / Math.max(den, 1e-300); });
  }

  /* Per-trajectory phase randomization of the deviations from the mean (same phases across dims). */
  function phaseRandomize(trajs, L, D, rnd) {
    const mean = meanOf(trajs);
    const half = Math.floor(L / 2);
    const cosT = new Float64Array(L), sinT = new Float64Array(L);
    for (let i = 0; i < L; i++) {
      cosT[i] = Math.cos((2 * Math.PI * i) / L);
      sinT[i] = Math.sin((2 * Math.PI * i) / L);
    }
    return trajs.map(function (x) {
      const out = new Float64Array(L * D);
      const phases = new Float64Array(half + 1);
      for (let f = 1; f <= half; f++) phases[f] = L % 2 === 0 && f === half ? 0 : rnd() * 2 * Math.PI;
      const re = new Float64Array(half + 1), im = new Float64Array(half + 1);
      for (let d = 0; d < D; d++) {
        for (let f = 0; f <= half; f++) {
          let a = 0, b = 0;
          for (let t = 0; t < L; t++) {
            const v = x[t * D + d] - mean[t * D + d];
            const idx = (f * t) % L;
            a += v * cosT[idx];
            b -= v * sinT[idx];
          }
          const c = Math.cos(phases[f]), s = Math.sin(phases[f]);
          re[f] = a * c - b * s;
          im[f] = a * s + b * c;
        }
        for (let t = 0; t < L; t++) {
          let v = re[0];
          for (let f = 1; f <= half; f++) {
            const idx = (f * t) % L;
            const w = L % 2 === 0 && f === half ? 1 : 2;
            v += w * (re[f] * cosT[idx] - im[f] * sinT[idx]);
          }
          out[t * D + d] = mean[t * D + d] + v / L;
        }
      }
      return out;
    });
  }

  /* Dynamic time warping. Returns h with x(h[j]) matched to ref[j]. */
  function dtwWarp(x, ref, L, D, band) {
    const acc = new Float64Array(L * L).fill(Infinity);
    for (let j = 0; j < L; j++) {
      const lo = Math.max(0, j - band), hi = Math.min(L - 1, j + band);
      for (let i = lo; i <= hi; i++) {
        let cost = 0;
        for (let d = 0; d < D; d++) {
          const diff = ref[j * D + d] - x[i * D + d];
          cost += diff * diff;
        }
        if (i === 0 && j === 0) { acc[0] = cost; continue; }
        let best = Infinity;
        if (i > 0) best = acc[j * L + i - 1];
        if (j > 0) {
          const up = acc[(j - 1) * L + i];
          if (up < best) best = up;
          if (i > 0) {
            const diag = acc[(j - 1) * L + i - 1];
            if (diag < best) best = diag;
          }
        }
        acc[j * L + i] = cost + best;
      }
    }
    const sums = new Float64Array(L), counts = new Float64Array(L);
    let j = L - 1, i = L - 1;
    for (;;) {
      sums[j] += i; counts[j] += 1;
      if (i === 0 && j === 0) break;
      let best = Infinity, nj = j, ni = i;
      if (i > 0 && j > 0 && acc[(j - 1) * L + i - 1] < best) { best = acc[(j - 1) * L + i - 1]; nj = j - 1; ni = i - 1; }
      if (j > 0 && acc[(j - 1) * L + i] < best) { best = acc[(j - 1) * L + i]; nj = j - 1; ni = i; }
      if (i > 0 && acc[j * L + i - 1] < best) { best = acc[j * L + i - 1]; nj = j; ni = i - 1; }
      j = nj; i = ni;
    }
    const h = new Float64Array(L);
    for (let k = 0; k < L; k++) h[k] = sums[k] / counts[k];
    return h;
  }

  function warp(x, h, L, D) {
    const out = new Float64Array(L * D);
    for (let j = 0; j < L; j++) {
      const pos = Math.min(Math.max(h[j], 0), L - 1);
      const i0 = Math.min(Math.floor(pos), L - 1), i1 = Math.min(i0 + 1, L - 1), w = pos - i0;
      for (let d = 0; d < D; d++) out[j * D + d] = (1 - w) * x[i0 * D + d] + w * x[i1 * D + d];
    }
    return out;
  }

  /* Inverse of `warp`: place an aligned trajectory back on the demo's own time axis. */
  function unwarp(aligned, h, L, D) {
    const hm = new Float64Array(L);
    let run = -Infinity;
    for (let j = 0; j < L; j++) {
      run = Math.max(run, Math.min(Math.max(h[j], 0), L - 1));
      hm[j] = run + 1e-6 * j;
    }
    const g = new Float64Array(L);
    let seg = 0;
    for (let s = 0; s < L; s++) {
      if (s <= hm[0]) { g[s] = 0; continue; }
      if (s >= hm[L - 1]) { g[s] = L - 1; continue; }
      while (seg < L - 2 && hm[seg + 1] < s) seg++;
      g[s] = seg + (s - hm[seg]) / (hm[seg + 1] - hm[seg]);
    }
    return warp(aligned, g, L, D);
  }

  function alignSet(trajs, L, D, band, iters) {
    let ref = meanOf(trajs);
    for (let it = 0; it < iters; it++) {
      ref = meanOf(trajs.map(function (x) { return warp(x, dtwWarp(x, ref, L, D, band), L, D); }));
    }
    return { ref: ref, H: trajs.map(function (x) { return dtwWarp(x, ref, L, D, band); }) };
  }

  /* Held-out residual of "shared path + exact time warp + k shape coordinates", in the demo's own time. */
  function alignedHeldout(trajs, L, D, band, iters, kmax, seed, folds) {
    folds = folds || 5;
    const N = trajs.length;
    const order = makeRng(seed).permutation(N);
    const num = new Float64Array(kmax + 1);
    let den = 0;
    for (let f = 0; f < folds; f++) {
      const test = [], train = [];
      order.forEach(function (idx, pos) { (pos % folds === f ? test : train).push(trajs[idx]); });
      if (!test.length || train.length < 2) continue;
      const muLin = meanOf(train);
      const al = alignSet(train, L, D, band, iters);
      const model = pca(train.map(function (x, i) { return warp(x, al.H[i], L, D); }), kmax);
      for (const x of test) {
        for (let m = 0; m < x.length; m++) den += (x[m] - muLin[m]) * (x[m] - muLin[m]);
        const h = dtwWarp(x, al.ref, L, D, band);
        const a = warp(x, h, L, D);
        const hat = Float64Array.from(model.mean);
        for (let k = 0; k <= kmax; k++) {
          if (k > 0 && k - 1 < model.comps.length) {
            const v = model.comps[k - 1];
            let c = 0;
            for (let m = 0; m < a.length; m++) c += (a[m] - model.mean[m]) * v[m];
            for (let m = 0; m < a.length; m++) hat[m] += c * v[m];
          }
          const rec = unwarp(hat, h, L, D);
          let err = 0;
          for (let m = 0; m < x.length; m++) err += (x[m] - rec[m]) * (x[m] - rec[m]);
          num[k] += err;
        }
      }
    }
    return Float64Array.from(num, function (v) { return v / Math.max(den, 1e-300); });
  }

  /* Full analysis of one set of trajectories (already resampled and standardized). */
  function analyzeTrajectories(trajs, L, D, opts) {
    const kmax = Math.min(opts.kmax, trajs.length - 1);
    const rnd = makeRng(opts.seed + 1);
    const out = { n: trajs.length, kmax: kmax };
    out.model = pca(trajs, Math.max(kmax, 12));
    out.evr = evrOf(out.model);
    out.heldout = heldoutCurve(trajs, kmax, opts.seed);
    out.sigma = Float64Array.from(out.model.evals, function (v) { return Math.sqrt(v / Math.max(trajs.length - 1, 1)); });
    out.z = trajs.map(function (x) {
      const r = new Float64Array(x.length);
      for (let m = 0; m < x.length; m++) r[m] = x[m] - out.model.mean[m];
      return out.model.comps.map(function (v) { return dot(r, v); });
    });
    const surrogate = phaseRandomize(trajs, L, D, rnd);
    out.evrSurrogate = evrOf(pca(surrogate, kmax));
    out.heldoutSurrogate = heldoutCurve(surrogate, kmax, opts.seed);
    if (opts.align) {
      out.alignedHeldout = alignedHeldout(trajs, L, D, opts.band, opts.alignIters, kmax, opts.seed);
      out.alignedHeldoutSurrogate = alignedHeldout(surrogate, L, D, opts.band, opts.alignIters, kmax, opts.seed);
      const al = alignSet(trajs, L, D, opts.band, opts.alignIters);
      out.H = al.H;
      out.alignedModel = pca(trajs.map(function (x, i) { return warp(x, al.H[i], L, D); }), Math.max(kmax, 12));
      out.alignedSigma = Float64Array.from(out.alignedModel.evals, function (v) { return Math.sqrt(v / Math.max(trajs.length - 1, 1)); });
    }
    return out;
  }

  /* Sample a trajectory from "mean + top-k subspace"; z is in units of each component's std. */
  function sampleLinear(model, sigma, z) {
    const x = Float64Array.from(model.mean);
    for (let k = 0; k < z.length && k < model.comps.length; k++) {
      const w = z[k] * sigma[k], v = model.comps[k];
      for (let m = 0; m < x.length; m++) x[m] += w * v[m];
    }
    return x;
  }

  /* Chunk-level view: PCA of all `horizon`-step chunks, coloured by phase in the episode. */
  function analyzeChunks(signals, std, horizon, maxChunks, bins) {
    const D = signals[0].D;
    let count = 0;
    for (const s of signals) count += Math.max(s.T - horizon + 1, 0);
    const stride = Math.max(1, Math.ceil(count / maxChunks));
    const chunks = [], phase = [], owner = [];
    signals.forEach(function (s, idx) {
      for (let t = 0; t + horizon <= s.T; t += stride) {
        const c = new Float64Array(horizon * D);
        for (let k = 0; k < horizon; k++) for (let d = 0; d < D; d++) c[k * D + d] = s.data[(t + k) * D + d] / std[d];
        chunks.push(c);
        phase.push(t / Math.max(s.T - horizon, 1));
        owner.push(idx);
      }
    });
    if (chunks.length < 3) return null;
    const model = pca(chunks, 2);
    const evr = evrOf(model);
    const coords = chunks.map(function (c) {
      const r = new Float64Array(c.length);
      for (let m = 0; m < c.length; m++) r[m] = c[m] - model.mean[m];
      return [dot(r, model.comps[0]), dot(r, model.comps[1])];
    });
    const sums = [], counts = new Float64Array(bins);
    for (let b = 0; b < bins; b++) sums.push(new Float64Array(horizon * D));
    const binOf = phase.map(function (p) { return Math.min(Math.floor(p * bins), bins - 1); });
    chunks.forEach(function (c, i) {
      counts[binOf[i]]++;
      const s = sums[binOf[i]];
      for (let m = 0; m < c.length; m++) s[m] += c[m];
    });
    let within = 0;
    chunks.forEach(function (c, i) {
      const s = sums[binOf[i]], n = counts[binOf[i]];
      for (let m = 0; m < c.length; m++) within += (c[m] - s[m] / n) * (c[m] - s[m] / n);
    });
    return {
      coords: coords, phase: phase, owner: owner, evr: evr, stride: stride, count: chunks.length,
      k90: kAt(evr, 0.9), pr: participationRatio(model),
      phaseExplained: 1 - within / Math.max(model.total, 1e-300),
    };
  }

  /* Rule-based event segmentation of one episode in its own time.
   * Boundaries are (a) jumps of two-valued dims such as a gripper command and (b) the edges of pauses,
   * i.e. stretches where the normalized speed stays low. `isVel[d]` marks dims that already hold per-step
   * changes (delta actions); the other continuous dims are differenced to get a rate. */
  function segmentEvents(sig, isVel) {
    const T = sig.T, D = sig.D, x = sig.data;
    const binary = [], cont = [], mids = [];
    for (let d = 0; d < D; d++) {
      let lo = Infinity, hi = -Infinity;
      for (let t = 0; t < T; t++) { const v = x[t * D + d]; if (v < lo) lo = v; if (v > hi) hi = v; }
      const range = hi - lo;
      if (!(range > 1e-9)) continue;
      let near = 0;
      for (let t = 0; t < T; t++) { const v = x[t * D + d]; if (v - lo < 0.1 * range || hi - v < 0.1 * range) near++; }
      if (near >= 0.95 * T) { binary.push(d); mids.push((lo + hi) / 2); } else cont.push(d);
    }
    const levels = binary.map(function (d, i) {
      const out = new Uint8Array(T);
      for (let t = 0; t < T; t++) out[t] = x[t * D + d] > mids[i] ? 1 : 0;
      return out;
    });
    const raw = new Float64Array(T);
    for (const d of cont) {
      const r = new Float64Array(T);
      let ms = 0;
      for (let t = 0; t < T; t++) {
        r[t] = isVel && isVel[d] ? x[t * D + d] : (t + 1 < T ? x[(t + 1) * D + d] - x[t * D + d] : (T > 1 ? x[(T - 1) * D + d] - x[(T - 2) * D + d] : 0));
        ms += r[t] * r[t];
      }
      ms /= T;
      if (ms > 0) for (let t = 0; t < T; t++) raw[t] += (r[t] * r[t]) / ms;
    }
    const half = Math.max(1, Math.round(T / 100));
    const speed = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      let s = 0, n = 0;
      for (let k = Math.max(0, t - half); k <= Math.min(T - 1, t + half); k++) { s += Math.sqrt(raw[k] / Math.max(cont.length, 1)); n++; }
      speed[t] = s / n;
    }
    const sorted = Float64Array.from(speed).sort();
    const threshold = 0.2 * sorted[Math.floor(0.75 * (T - 1))];
    const minPause = Math.max(4, Math.round(T * 0.03));
    const pause = new Uint8Array(T);
    for (let t = 0; t < T;) {
      if (speed[t] >= threshold) { t++; continue; }
      let e = t;
      while (e + 1 < T && speed[e + 1] < threshold) e++;
      if (e - t + 1 >= minPause) for (let k = t; k <= e; k++) pause[k] = 1;
      t = e + 1;
    }
    const toggles = [], cuts = [];
    for (let t = 1; t < T; t++) {
      let jump = false;
      for (const lv of levels) if (lv[t] !== lv[t - 1]) jump = true;
      if (jump) toggles.push(t);
      if (jump || pause[t] !== pause[t - 1]) cuts.push(t);
    }
    const kept = [];
    let last = 0;
    for (const c of cuts) if (c - last >= 3 && T - c >= 3) { kept.push(c); last = c; }
    const events = [];
    let start = 0;
    for (const c of kept.concat([T])) {
      let p = 0;
      for (let t = start; t < c; t++) p += pause[t];
      const mid = Math.min(Math.floor((start + c) / 2), T - 1);
      events.push({ start: start, end: c - 1, kind: p > (c - start) / 2 ? 'pause' : 'move', levels: levels.map(function (lv) { return lv[mid]; }) });
      start = c;
    }
    return { speed: speed, binary: binary, cont: cont, levels: levels, toggles: toggles, cuts: kept, events: events };
  }

  /* ---------- manifold shape: point clouds, neighbour graphs, nonlinear embeddings ---------- */

  /* One point per `horizon`-step window; windows of one episode stay in order so they can be drawn as a curve. */
  function chunkCloud(signals, std, horizon, maxPoints) {
    const D = signals[0].D;
    let count = 0;
    for (const s of signals) count += Math.max(s.T - horizon + 1, 0);
    const stride = Math.max(1, Math.ceil(count / maxPoints));
    const points = [], phase = [], owner = [];
    signals.forEach(function (s, idx) {
      for (let t = 0; t + horizon <= s.T; t += stride) {
        const c = new Float64Array(horizon * D);
        for (let k = 0; k < horizon; k++) for (let d = 0; d < D; d++) c[k * D + d] = s.data[(t + k) * D + d] / std[d];
        points.push(c); phase.push(t / Math.max(s.T - horizon, 1)); owner.push(idx);
      }
    });
    return { points: points, phase: phase, owner: owner, stride: stride };
  }

  function distanceMatrix(points) {
    const n = points.length, M = points[0].length, out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      const a = points[i];
      for (let j = i + 1; j < n; j++) {
        const b = points[j];
        let s = 0;
        for (let m = 0; m < M; m++) { const d = a[m] - b[m]; s += d * d; }
        out[i * n + j] = out[j * n + i] = Math.sqrt(s);
      }
    }
    return out;
  }

  /* TwoNN intrinsic dimension (Facco et al. 2017): uses only the two nearest neighbours of each point. */
  function twoNN(dist, n) {
    let sum = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      let r1 = Infinity, r2 = Infinity;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const d = dist[i * n + j];
        if (d < r1) { r2 = r1; r1 = d; } else if (d < r2) r2 = d;
      }
      if (r1 > 0 && r2 > r1 && isFinite(r2)) { sum += Math.log(r2 / r1); cnt++; }
    }
    return cnt / Math.max(sum, 1e-12);
  }

  function nearestK(dist, n, i, k) {
    const idx = new Int32Array(k).fill(-1), val = new Float64Array(k).fill(Infinity);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = dist[i * n + j];
      if (d >= val[k - 1]) continue;
      let pos = k - 1;
      while (pos > 0 && val[pos - 1] > d) { val[pos] = val[pos - 1]; idx[pos] = idx[pos - 1]; pos--; }
      val[pos] = d; idx[pos] = j;
    }
    return { idx: idx, val: val };
  }

  /* Shortest-path distances on the symmetrized k-nearest-neighbour graph (Dijkstra from every point).
   * `chains[i]` names the curve a point belongs to; consecutive points of one curve are also linked. */
  function geodesics(dist, n, k, chains) {
    k = Math.min(k, n - 1);
    const nbr = Array.from({ length: n }, function () { return new Map(); });
    const kth = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const nn = nearestK(dist, n, i, k);
      kth[i] = nn.val[k - 1];
      for (let q = 0; q < k; q++) if (nn.idx[q] >= 0) { nbr[i].set(nn.idx[q], nn.val[q]); nbr[nn.idx[q]].set(i, nn.val[q]); }
    }
    if (chains) for (let i = 1; i < n; i++) if (chains[i] === chains[i - 1]) { nbr[i].set(i - 1, dist[i * n + i - 1]); nbr[i - 1].set(i, dist[i * n + i - 1]); }
    const adjJ = nbr.map(function (m) { return Int32Array.from(m.keys()); });
    const adjW = nbr.map(function (m) { return Float64Array.from(m.values()); });
    const geo = new Float64Array(n * n).fill(Infinity);
    let cap = 16;
    for (const a of adjJ) cap += a.length;
    const hd = new Float64Array(cap), hn = new Int32Array(cap);
    let connected = true;
    for (let s = 0; s < n; s++) {
      const base = s * n;
      geo[base + s] = 0;
      let size = 0;
      hd[0] = 0; hn[0] = s; size = 1;
      while (size > 0) {
        const d = hd[0], u = hn[0];
        size--;
        if (size > 0) {
          const ld = hd[size], ln = hn[size];
          let pos = 0;
          for (;;) {
            let c = 2 * pos + 1;
            if (c >= size) break;
            if (c + 1 < size && hd[c + 1] < hd[c]) c++;
            if (hd[c] >= ld) break;
            hd[pos] = hd[c]; hn[pos] = hn[c]; pos = c;
          }
          hd[pos] = ld; hn[pos] = ln;
        }
        if (d > geo[base + u]) continue;
        const js = adjJ[u], ws = adjW[u];
        for (let e = 0; e < js.length; e++) {
          const v = js[e], nd = d + ws[e];
          if (nd < geo[base + v]) {
            geo[base + v] = nd;
            let pos = size++;
            while (pos > 0) {
              const par = (pos - 1) >> 1;
              if (hd[par] <= nd) break;
              hd[pos] = hd[par]; hn[pos] = hn[par]; pos = par;
            }
            hd[pos] = nd; hn[pos] = v;
          }
        }
      }
      if (connected) for (let v = 0; v < n; v++) if (geo[base + v] === Infinity) { connected = false; break; }
    }
    return { geo: geo, connected: connected, kth: kth };
  }

  function medianOf(values) {
    const a = Float64Array.from(values).sort();
    return a.length ? a[a.length >> 1] : NaN;
  }

  /* Median of geodesic / Euclidean distance over the farther half of all pairs. 1 for a flat convex set. */
  function geodesicRatio(dist, geo, n) {
    const ds = [], rs = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const d = dist[i * n + j], g = geo[i * n + j];
      if (d > 0 && isFinite(g)) { ds.push(d); rs.push(g / d); }
    }
    const cut = medianOf(ds), far = [];
    for (let q = 0; q < ds.length; q++) if (ds[q] > cut) far.push(rs[q]);
    return medianOf(far);
  }

  /* Midpoint test for convexity: distance from the midpoint of two points to the nearest other point,
   * in units of the typical nearest-neighbour distance. */
  function midpointRatio(points, dist, n, rnd, pairs) {
    const M = points[0].length, nn = new Float64Array(n).fill(Infinity);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (j !== i && dist[i * n + j] < nn[i]) nn[i] = dist[i * n + j];
    const typical = medianOf(nn), mids = [], mid = new Float64Array(M);
    for (let q = 0; q < pairs; q++) {
      const a = Math.floor(rnd() * n);
      let b = Math.floor(rnd() * (n - 1));
      if (b >= a) b++;
      for (let m = 0; m < M; m++) mid[m] = (points[a][m] + points[b][m]) / 2;
      let best = Infinity;
      for (let i = 0; i < n; i++) {
        if (i === a || i === b) continue;
        const x = points[i];
        let s = 0;
        for (let m = 0; m < M && s < best; m++) { const d = x[m] - mid[m]; s += d * d; }
        if (s < best) best = s;
      }
      mids.push(Math.sqrt(best));
    }
    return medianOf(mids) / Math.max(typical, 1e-12);
  }

  /* A Gaussian with the same mean and covariance as the points: the convex ellipsoid they are compared with. */
  function gaussianLike(points, rnd) {
    const n = points.length, M = points[0].length;
    const model = pca(points, Math.min(n - 1, M, 64));
    return points.map(function () {
      const x = Float64Array.from(model.mean);
      for (let k = 0; k < model.comps.length; k++) {
        const w = rnd.normal() * Math.sqrt(model.evals[k] / Math.max(n - 1, 1)), v = model.comps[k];
        for (let m = 0; m < M; m++) x[m] += w * v[m];
      }
      return x;
    });
  }

  /* Midpoint test inside each group (e.g. phase bins), each with its own Gaussian control. */
  function midpointWithinGroups(points, groups, rnd) {
    const byGroup = new Map();
    groups.forEach(function (g, i) { if (!byGroup.has(g)) byGroup.set(g, []); byGroup.get(g).push(points[i]); });
    let value = 0, nul = 0, used = 0;
    for (const members of byGroup.values()) {
      if (members.length < 12) continue;
      value += midpointRatio(members, distanceMatrix(members), members.length, rnd, 120);
      const g = gaussianLike(members, rnd);
      nul += midpointRatio(g, distanceMatrix(g), g.length, rnd, 120);
      used++;
    }
    return used ? { value: value / used, nul: nul / used } : null;
  }

  /* All shape diagnostics for one point cloud, with the same numbers for its Gaussian control. */
  function shapeMetrics(points, rnd, k, groups, chains) {
    const n = points.length;
    const dist = distanceMatrix(points), g = geodesics(dist, n, k, chains);
    const out = {
      n: n, dist: dist, geo: g.geo, kth: g.kth, connected: g.connected,
      dim: twoNN(dist, n), geoRatio: geodesicRatio(dist, g.geo, n), mid: midpointRatio(points, dist, n, rnd, 500),
    };
    const nul = gaussianLike(points, rnd), nd = distanceMatrix(nul), ng = geodesics(nd, n, k);
    out.nullDim = twoNN(nd, n); out.nullGeoRatio = geodesicRatio(nd, ng.geo, n); out.nullMid = midpointRatio(nul, nd, n, rnd, 500);
    out.within = groups ? midpointWithinGroups(points, groups, rnd) : null;
    return out;
  }

  function embedPCA(points) {
    const model = pca(points, 3);
    return points.map(function (x) {
      const r = new Float64Array(x.length);
      for (let m = 0; m < x.length; m++) r[m] = x[m] - model.mean[m];
      return [0, 1, 2].map(function (c) { return c < model.comps.length ? dot(r, model.comps[c]) : 0; });
    });
  }

  /* Isomap: classical multidimensional scaling of the graph geodesic distances. */
  function embedIsomap(geo, n) {
    let maxFinite = 0;
    for (let i = 0; i < n * n; i++) if (isFinite(geo[i]) && geo[i] > maxFinite) maxFinite = geo[i];
    const B = new Float64Array(n * n), rowMean = new Float64Array(n);
    let grand = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const g = isFinite(geo[i * n + j]) ? geo[i * n + j] : maxFinite, v = g * g;
      B[i * n + j] = v; rowMean[i] += v / n; grand += v / (n * n);
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i * n + j] = -0.5 * (B[i * n + j] - rowMean[i] - rowMean[j] + grand);
    const e = eigTop(B, n, 10);
    return Array.from({ length: n }, function (_, i) {
      return [0, 1, 2].map(function (c) { return e.values[c] > 0 ? e.vectors[c][i] * Math.sqrt(e.values[c]) : 0; });
    });
  }

  /* Diffusion map with density normalization (alpha = 1). The kernel width is local: the product of the
   * two points' k-th neighbour distances (self-tuning, Zelnik-Manor & Perona 2004). */
  function embedDiffusion(dist, kth, n) {
    const floor = medianOf(kth) * 0.1 || 1e-9;
    const K = new Float64Array(n * n), q = new Float64Array(n), deg = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = Math.exp(-(dist[i * n + j] * dist[i * n + j]) / (4 * Math.max(kth[i], floor) * Math.max(kth[j], floor)));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q[i] += K[i * n + j];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { K[i * n + j] /= q[i] * q[j]; deg[i] += K[i * n + j]; }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] /= Math.sqrt(deg[i] * deg[j]);
    const e = eigTop(K, n, 10);
    return Array.from({ length: n }, function (_, i) {
      const v0 = e.vectors[0][i] || 1e-12;
      return [1, 2, 3].map(function (c) { return c < e.vectors.length ? (e.values[c] * e.vectors[c][i]) / v0 : 0; });
    });
  }

  /* Convex hull of 2-D points (monotone chain). Returns the hull vertices in order. */
  function convexHull2D(pts) {
    const p = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    if (p.length < 3) return p;
    const cross = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
    const lower = [], upper = [];
    for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  const api = {
    chunkCloud: chunkCloud, distanceMatrix: distanceMatrix, twoNN: twoNN, geodesics: geodesics, geodesicRatio: geodesicRatio,
    midpointRatio: midpointRatio, gaussianLike: gaussianLike, shapeMetrics: shapeMetrics, embedPCA: embedPCA,
    embedIsomap: embedIsomap, embedDiffusion: embedDiffusion, convexHull2D: convexHull2D,
    segmentEvents: segmentEvents,
    makeRng: makeRng, resample: resample, buildSignal: buildSignal, jacobi: jacobi, eigTop: eigTop, meanOf: meanOf,
    pca: pca, evrOf: evrOf, kAt: kAt, participationRatio: participationRatio, heldoutCurve: heldoutCurve,
    phaseRandomize: phaseRandomize, dtwWarp: dtwWarp, warp: warp, unwarp: unwarp, alignSet: alignSet,
    alignedHeldout: alignedHeldout, analyzeTrajectories: analyzeTrajectories, sampleLinear: sampleLinear,
    analyzeChunks: analyzeChunks, dot: dot,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ActionCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
