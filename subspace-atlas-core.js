/* Subspace Atlas numeric API (UMD; requires ActionCore).
 * unpack accepts packed int16/base64 or expanded {fields,groups,episodes} JSON.
 * Expanded episodes: {id,group,T,data:{field:flat or T*D numeric arrays},context?}.
 * Optional context is an INITIAL observation vector, never action/future data.
 * Packed datasets can put observation vectors in contexts[episodeIndex].
 *
 * analyze(ds,{group:0,signal:'action',length:64,seed:42,maxRank:15,context:'eepos'})
 * fits ONE task, splitting demonstrations BEFORE all fitting, 60/20/20 (rounded
 * down for train/validation). At least 15 demonstrations required. This is a
 * random demonstration holdout, NOT an official LIBERO benchmark split.
 * rows: physical signals; normalized: train-channel standardized signals.
 * model: normalized trajectory mean, orthonormal comps, eigenvalues, score sd,
 * physical channel shift/scale. coords: unwhitened PCA scores for ALL episodes.
 * split indices address rows/episodes; sourceIndices address input episodes.
 * curves arrays index K=0..maxRank. Values=sum SSE/sum error-to-training-mean.
 * Zero denominator => null. Ratios may exceed one and are NEVER clipped to one.
 * Rank uses only validation ORACLE reconstruction: smallest K <=10% residual,
 * else minimum validation error. This is representation rank, NOT policy rank.
 * Prediction: train-only standardized initial context -> ridge with unpenalized
 * intercept -> scores. Alpha is validation-selected at the recommended rank.
 * Shuffled control permutes train context and independently selects alpha.
 * Error identity: prediction SSE = orthogonal residual SSE + score error SSE.
 * testScores/metrics report selectedRank. decode(result,coords) returns physical
 * flattened trajectory; pass truncated coords for a lower-rank reconstruction.
 * No DTW, no claim of rollout success, no future data in context prediction.
 */
(function (root) {
  'use strict';
  const Core = typeof module !== 'undefined' && module.exports ? require('./atlas-numeric-core.js') : root.ActionCore;
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const integer = (v, min) => Number.isInteger(v) && v >= min;
  function fail(s) { throw new Error(s); }
  function vector(v, size, label) {
    if (!Array.isArray(v) && !ArrayBuffer.isView(v)) fail(label + ': expected numeric array');
    const a = Array.isArray(v) && v.length && Array.isArray(v[0]) ? v.flat() : v;
    if (a.length !== size) fail(label + ': wrong number of values (expected ' + size + ')');
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) { if (!finite(a[i])) fail(label + ': non-finite or non-numeric value'); out[i] = a[i]; }
    return out;
  }
  function unpack(input) {
    let j = input;
    if (typeof j === 'string') { try { j = JSON.parse(j); } catch (_) { fail('Invalid JSON'); } }
    if (!j || typeof j !== 'object') fail('Expected dataset object');
    if (!Array.isArray(j.fields) || !j.fields.length) fail('fields must be nonempty');
    if (!Array.isArray(j.groups) || !j.groups.length || j.groups.some(g => typeof g !== 'string')) fail('groups must contain task names');
    if (!Array.isArray(j.episodes) || !j.episodes.length) fail('episodes must be nonempty');
    const names = new Set();
    const fields = j.fields.map(f => {
      if (!f || typeof f.name !== 'string' || !f.name.length || names.has(f.name)) fail('Invalid or duplicate field name');
      if (!integer(f.dim, 1) || f.dim > 1024) fail('Invalid field dimension');
      names.add(f.name); return { name: f.name, dim: f.dim };
    });
    const packed = Array.isArray(j.episodes[0]);
    let total = 0;
    const episodes = j.episodes.map((r, i) => {
      if (packed && (!Array.isArray(r) || r.length < 3)) fail('Malformed packed episode');
      const e = packed ? { id: r[0], group: r[1], T: r[2] } : r;
      if (!e || !integer(e.T, 1) || e.T > 1000000) fail('Episode ' + i + ': invalid T');
      if (!integer(e.group, 0) || e.group >= j.groups.length) fail('Episode ' + i + ': invalid group');
      if (e.id != null && typeof e.id !== 'string' && !finite(e.id)) fail('Episode ' + i + ': invalid id');
      total += e.T;
      const ep = { id: e.id == null ? i : e.id, group: e.group, T: e.T, data: Object.create(null) };
      const ctx = packed ? j.contexts && j.contexts[i] : e.context;
      if (ctx != null) {
        if (!integer(ctx.length, 1) || ctx.length > 4096) fail('Invalid initial context vector');
        ep.context = vector(ctx, ctx.length, 'Initial context ' + i);
      }
      if (e.contextProvenance) ep.contextProvenance = e.contextProvenance;
      // Display-only source image references never enter numerical features.
      if (e.images) ep.images = e.images;
      return ep;
    });
    if (total * fields.reduce((s, f) => s + f.dim, 0) > 50000000) fail('Dataset exceeds 50 million numeric values');
    if (packed) {
      if (!j.data || typeof j.data !== 'object') fail('Missing packed data');
      fields.forEach((f, fi) => {
        const scale = j.fields[fi].scale, encoded = j.data[f.name];
        if (!finite(scale) || scale <= 0) fail(f.name + ': scale must be positive');
        if (typeof encoded !== 'string' || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(encoded)) fail(f.name + ': invalid base64');
        const bytes = typeof Buffer !== 'undefined' ? Uint8Array.from(Buffer.from(encoded, 'base64')) : Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
        if (bytes.length !== total * f.dim * 2) fail(f.name + ': packed length does not match episodes');
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        let offset = 0;
        episodes.forEach(ep => {
          const a = new Float64Array(ep.T * f.dim);
          for (let i = 0; i < a.length; i++) { a[i] = view.getInt16(offset, true) * scale; offset += 2; if (!finite(a[i])) fail('Decoded numeric overflow'); }
          ep.data[f.name] = a;
        });
      });
    } else episodes.forEach((ep, i) => {
      if (!j.episodes[i].data) fail('Episode ' + i + ': missing data');
      fields.forEach(f => { ep.data[f.name] = vector(j.episodes[i].data[f.name], ep.T * f.dim, 'Episode ' + i + ' ' + f.name); });
    });
    return { name: typeof j.name === 'string' ? j.name : 'Uploaded dataset', fields, groups: j.groups.slice(), episodes, contextInfo: j.contextInfo || null };
  }
  function splitIndices(n, seed) {
    if (!integer(n, 1)) fail('Invalid episode count');
    const p = Core.makeRng(seed == null ? 42 : seed).permutation(n), a = Math.floor(.6 * n), b = Math.floor(.2 * n);
    return { train: p.slice(0, a), validation: p.slice(a, a + b), test: p.slice(a + b) };
  }
  function signalInfo(fields, signal) {
    signal = signal || 'action';
    if (!['action', 'action_xyz', 'eepos', 'relative_eepos', 'cumaction'].includes(signal)) fail('Unknown signal: ' + signal);
    const field = signal === 'eepos' || signal === 'relative_eepos' ? 'eepos' : 'action', spec = fields.find(f => f.name === field);
    if (!spec) fail('Dataset does not contain ' + field);
    const D = signal === 'action' ? spec.dim : 3;
    if (spec.dim < D) fail(field + ' requires at least ' + D + ' channels');
    const labels = field === 'eepos' ? ['EE x', 'EE y', 'EE z'] : Array.from({ length: D }, (_, i) => ['action x', 'action y', 'action z', 'rotation x', 'rotation y', 'rotation z', 'gripper'][i] || 'action ' + i);
    return { D, labels, field, dims: Array.from({ length: D }, (_, i) => i), kind: signal };
  }
  function signalRow(ep, fields, signal, length) {
    const info = signalInfo(fields, signal), dim = fields.find(f => f.name === info.field).dim;
    const input = ep.data[info.field], a = new Float64Array(ep.T * info.D), acc = new Float64Array(info.D);
    for (let t = 0; t < ep.T; t++) for (let d = 0; d < info.D; d++) {
      let v = input[t * dim + d];
      if (!finite(v)) fail('Non-finite signal');
      if (signal === 'relative_eepos') v -= input[d];
      // Prefix sums are controller-command integrals, not physical EE paths.
      if (signal === 'cumaction') { acc[d] += v; v = acc[d]; }
      a[t * info.D + d] = v;
    }
    return Core.resample(a, ep.T, info.D, length == null ? 64 : length);
  }
  function channelScale(rows, train, D) {
    const shift = new Float64Array(D), scale = new Float64Array(D), count = train.length * rows[0].length / D, constantChannels = [];
    train.forEach(i => { for (let j = 0; j < rows[i].length; j++) shift[j % D] += rows[i][j] / count; });
    train.forEach(i => { for (let j = 0; j < rows[i].length; j++) scale[j % D] += (rows[i][j] - shift[j % D]) ** 2 / count; });
    for (let d = 0; d < D; d++) { scale[d] = Math.sqrt(scale[d]); if (scale[d] <= 1e-12 * Math.max(1, Math.abs(shift[d]))) { constantChannels.push(d); scale[d] = 1; } }
    return { shift, scale, constantChannels };
  }
  function project(row, model) { return Float64Array.from(model.comps, c => c.reduce((sum, v, j) => sum + (row[j] - model.mean[j]) * v, 0)); }
  function decode(result, coords) {
    const m = result.model, out = Float64Array.from(m.mean);
    for (let k = 0; k < Math.min(coords.length, m.comps.length); k++) {
      if (!finite(coords[k])) fail('Non-finite coordinate');
      for (let j = 0; j < out.length; j++) out[j] += coords[k] * m.comps[k][j];
    }
    for (let j = 0; j < out.length; j++) out[j] = out[j] * m.scale[j % result.D] + m.shift[j % result.D];
    return out;
  }
  // Cholesky factorization for symmetric positive definite ridge systems.
  function cholesky(A) {
    const n = A.length, L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) { if (!(s > 0)) fail('Non-positive ridge system'); L[i][j] = Math.sqrt(s); }
      else L[i][j] = s / L[j][j];
    }
    return L;
  }
  function solve(L, b) {
    const n = b.length, x = Float64Array.from(b);
    for (let i = 0; i < n; i++) { for (let j = 0; j < i; j++) x[i] -= L[i][j] * x[j]; x[i] /= L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { for (let j = i + 1; j < n; j++) x[i] -= L[j][i] * x[j]; x[i] /= L[i][i]; }
    return x;
  }
  function fitContext(episodes, fields, train, coords, seed, mode, info, validation, selectionRank) {
    let raw;
    if (mode === 'observation') {
      const p = episodes[0].context && episodes[0].context.length;
      if (!p || episodes.some(ep => !ep.context || ep.context.length !== p)) fail('Observation mode requires equal-length INITIAL context vectors for every episode');
      raw = episodes.map(ep => ep.context);
    } else {
      const f = fields.find(f => f.name === 'eepos');
      if (!f || f.dim < 3) return { available: false, mode, reason: 'Initial observed eepos XYZ is unavailable', alpha: null };
      raw = episodes.map(ep => ep.data.eepos.slice(0, 3));
    }
    const P = raw[0].length, scaling = channelScale(raw, train, P);
    const features = raw.map(r => Float64Array.from(r, (v, d) => (v - scaling.shift[d]) / scaling.scale[d]));
    const K = coords[0].length, perm = Core.makeRng((seed ^ 0x4f1bbcdc) >>> 0).permutation(train.length);
    const ymean = new Float64Array(K);
    train.forEach(i => { for (let k = 0; k < K; k++) ymean[k] += coords[i][k] / train.length; });
    function fit(shuffle, alpha) {
      const X = train.map((i, p) => features[shuffle ? train[perm[p]] : i]);
      // Dual solve is cheaper for image/context features wider than train count.
      const dual = P > train.length, n = dual ? train.length : P;
      const A = Array.from({ length: n }, () => new Float64Array(n));
      for (let a = 0; a < n; a++) for (let b = 0; b <= a; b++) {
        let s = a === b ? alpha : 0;
        if (dual) s += Core.dot(X[a], X[b]); else for (const x of X) s += x[a] * x[b];
        A[a][b] = A[b][a] = s;
      }
      const L = cholesky(A);
      return Array.from({ length: K }, (_, k) => {
        const rhs = new Float64Array(n);
        if (dual) train.forEach((i, p) => { rhs[p] = coords[i][k] - ymean[k]; });
        else train.forEach((i, p) => { for (let d = 0; d < P; d++) rhs[d] += X[p][d] * (coords[i][k] - ymean[k]); });
        const sol = solve(L, rhs);
        if (!dual) return sol;
        const w = new Float64Array(P);
        X.forEach((x, p) => { for (let d = 0; d < P; d++) w[d] += x[d] * sol[p]; });
        return w;
      });
    }
    // Select regularization after the representation rank, using validation only.
    // Both real and shuffled predictors use the same grid and selection criterion.
    // The large finite penalty gives a near-mean predictor without non-JSON infinities.
    const alphaGrid = [.1, 1, 10, 100, 1000, 1000000];
    function select(shuffle) {
      const scores = []; let best = null;
      for (const alpha of alphaGrid) {
        const weights = fit(shuffle, alpha); let sse = 0;
        for (const i of validation) for (let k = 0; k < selectionRank; k++) {
          const estimate = ymean[k] + Core.dot(weights[k], features[i]);
          sse += (coords[i][k] - estimate) ** 2;
        }
        scores.push({ alpha, sse });
        if (!best || sse <= best.sse + 1e-12 * Math.max(1, best.sse)) best = { alpha, sse, weights };
      }
      return { ...best, scores };
    }
    const chosen = select(false), control = select(true);
    const weights = chosen.weights, shuffledWeights = control.weights;
    const predict = ws => features.map(x => Float64Array.from(ws, (w, k) => ymean[k] + Core.dot(w, x)));
    return { available: true, mode, dimensions: P, label: mode === 'observation' ? 'Initial observation features (not semantic image embeddings)' : 'Initial observed end-effector XYZ only (limited environment proxy)',
      provenance: info || null, alpha: chosen.alpha, shuffledAlpha: control.alpha, alphaGrid, selectionRank,
      validationAlphaScores: chosen.scores, shuffledValidationAlphaScores: control.scores,
      alphaSelection: 'Minimum validation coefficient SSE at the validation-selected representation rank; prefer stronger penalty on ties',
      shift: scaling.shift, scale: scaling.scale, constantChannels: scaling.constantChannels,
      raw, features, weights, shuffledWeights, intercept: ymean, trainingPermutation: perm, predictedCoords: predict(weights), shuffledCoords: predict(shuffledWeights) };
  }
  function ratio(a, b) { return b > 1e-24 ? a / b : null; }
  function aggregate(indices, energy, coords, predicted, shuffled, K) {
    let baselineSSE = 0;
    const oracleSSE = new Float64Array(K + 1), coefficientSSE = new Float64Array(K + 1), controlSSE = new Float64Array(K + 1);
    indices.forEach(i => {
      baselineSSE += energy[i]; let projected = 0, error = 0, control = 0;
      for (let k = 0; k <= K; k++) {
        if (k > 0) { projected += coords[i][k - 1] ** 2; if (predicted) error += (coords[i][k - 1] - predicted[i][k - 1]) ** 2; if (shuffled) control += (coords[i][k - 1] - shuffled[i][k - 1]) ** 2; }
        oracleSSE[k] += Math.max(0, energy[i] - projected); coefficientSSE[k] += error; controlSSE[k] += control;
      }
    });
    return { baselineSSE, oracleSSE, coefficientSSE: predicted ? coefficientSSE : null,
      predictedSSE: predicted ? Float64Array.from(oracleSSE, (v, k) => v + coefficientSSE[k]) : null,
      shuffledSSE: shuffled ? Float64Array.from(oracleSSE, (v, k) => v + controlSSE[k]) : null,
      oracle: Array.from(oracleSSE, v => ratio(v, baselineSSE)),
      coefficient: predicted ? Array.from(coefficientSSE, v => ratio(v, baselineSSE)) : Array(K + 1).fill(null),
      predicted: predicted ? Array.from(oracleSSE, (v, k) => ratio(v + coefficientSSE[k], baselineSSE)) : Array(K + 1).fill(null),
      shuffled: shuffled ? Array.from(oracleSSE, (v, k) => ratio(v + controlSSE[k], baselineSSE)) : Array(K + 1).fill(null) };
  }
  function analyze(input, opts) {
    if (!Core) fail('Load ActionCore before SubspaceAtlasCore');
    const ds = unpack(input); opts = opts || {};
    let group = opts.group == null ? 0 : opts.group;
    if (typeof group === 'string') group = ds.groups.indexOf(group);
    if (!integer(group, 0) || group >= ds.groups.length) fail('Select exactly one valid task group');
    const length = opts.length == null ? 64 : opts.length, seed = opts.seed == null ? 42 : opts.seed;
    const requestedRank = opts.maxRank == null ? 15 : opts.maxRank, signal = opts.signal || 'action', mode = opts.context || 'eepos';
    if (!integer(length, 2) || length > 1024) fail('Length must be 2..1024');
    if (!integer(requestedRank, 0) || requestedRank > 100) fail('Maximum rank must be 0..100');
    if (!integer(seed, 0) || seed > 4294967295) fail('Seed must be uint32');
    if (!['eepos', 'observation'].includes(mode)) fail('Unknown context mode');
    const sourceIndices = ds.episodes.map((_, i) => i).filter(i => ds.episodes[i].group === group);
    if (sourceIndices.length < 15) fail('At least 15 demonstrations per task are required');
    const episodes = sourceIndices.map(i => ds.episodes[i]), info = signalInfo(ds.fields, signal), D = info.D;
    const split = splitIndices(episodes.length, seed), rows = episodes.map(ep => signalRow(ep, ds.fields, signal, length)), scaling = channelScale(rows, split.train, D);
    const normalized = rows.map(row => Float64Array.from(row, (v, j) => (v - scaling.shift[j % D]) / scaling.scale[j % D]));
    const rawModel = Core.pca(split.train.map(i => normalized[i]), Math.min(requestedRank, split.train.length - 1, D * length));
    const comps = [], eigenvalues = [];
    // Reorthogonalize to preserve the exact Pythagorean error identity.
    rawModel.comps.forEach((candidate, k) => {
      if (!(rawModel.total > 1e-24) || !(rawModel.evals[k] > 1e-12 * rawModel.total)) return;
      const c = Float64Array.from(candidate);
      for (let pass = 0; pass < 2; pass++) for (const prev of comps) { const dot = Core.dot(prev, c); for (let j = 0; j < c.length; j++) c[j] -= dot * prev[j]; }
      const norm = Math.sqrt(Core.dot(c, c)); if (norm < 1e-10) return;
      for (let j = 0; j < c.length; j++) c[j] /= norm;
      let largest = 0; for (let j = 1; j < c.length; j++) if (Math.abs(c[j]) > Math.abs(c[largest])) largest = j;
      if (c[largest] < 0) for (let j = 0; j < c.length; j++) c[j] *= -1;
      comps.push(c); eigenvalues.push(rawModel.evals[k]);
    });
    const K = comps.length, model = { mean: rawModel.mean, comps, eigenvalues: Float64Array.from(eigenvalues), evals: Float64Array.from(eigenvalues),
      sd: Float64Array.from(eigenvalues, v => Math.sqrt(v / Math.max(1, split.train.length - 1))), shift: scaling.shift, scale: scaling.scale,
      constantChannels: scaling.constantChannels, total: rawModel.total, n: split.train.length, exact: rawModel.exact };
    const coords = normalized.map(row => project(row, model)), energy = normalized.map(row => row.reduce((s, v, j) => s + (v - model.mean[j]) ** 2, 0));
    const rankCurve = aggregate(split.validation, energy, coords, null, null, K).oracle;
    let selectedRank = rankCurve.findIndex(v => v != null && v <= .10), selectionReason = 'First validation oracle residual at or below 10%';
    if (selectedRank < 0) { selectedRank = 0; if (rankCurve[0] == null) selectionReason = 'Zero-energy validation baseline: rank selection undefined'; else { selectionReason = '10% target not reached; minimum validation oracle residual'; for (let k = 1; k <= K; k++) if (rankCurve[k] < rankCurve[selectedRank] - 1e-12) selectedRank = k; } }
    const context = fitContext(episodes, ds.fields, split.train, coords, seed, mode, ds.contextInfo, split.validation, selectedRank);
    const predictedCoords = context.available ? context.predictedCoords : null, shuffledCoords = context.available ? context.shuffledCoords : null;
    const train = aggregate(split.train, energy, coords, predictedCoords, shuffledCoords, K), validation = aggregate(split.validation, energy, coords, predictedCoords, shuffledCoords, K), test = aggregate(split.test, energy, coords, predictedCoords, shuffledCoords, K);
    const evr = eigenvalues.map(v => ratio(v, rawModel.total)); let running = 0;
    const cumulative = [rawModel.total > 1e-24 ? 0 : null, ...evr.map(v => v == null ? null : (running += v))];
    const testScores = split.test.map(i => {
      const one = aggregate([i], energy, coords, predictedCoords, shuffledCoords, K), k = selectedRank; let direct = null;
      if (predictedCoords) { direct = 0; for (let j = 0; j < model.mean.length; j++) { let estimate = model.mean[j]; for (let a = 0; a < k; a++) estimate += model.comps[a][j] * predictedCoords[i][a]; direct += (normalized[i][j] - estimate) ** 2; } }
      return { index: i, sourceIndex: sourceIndices[i], id: episodes[i].id, baselineSSE: energy[i], oracle: one.oracle[k], predicted: one.predicted[k], shuffled: one.shuffled[k], coefficient: one.coefficient[k],
        oracleSSE: one.oracleSSE[k], predictedSSE: one.predictedSSE ? one.predictedSSE[k] : null, coefficientSSE: one.coefficientSSE ? one.coefficientSSE[k] : null, shuffledSSE: one.shuffledSSE ? one.shuffledSSE[k] : null,
        directPredictedSSE: direct, decompositionGap: direct == null ? null : direct - one.predictedSSE[k] };
    });
    const k = selectedRank;
    return { name: ds.name, group, groupName: ds.groups[group], signal, signalInfo: info, labels: info.labels, D, length, seed, episodes, sourceIndices, rows, normalized, split, model, coords, predictedCoords, shuffledCoords, context,
      maxRank: K, requestedRank, selectedRank, selectionReason, rankTarget: .10, evr, cumulative,
      curves: { train: train.oracle, validation: validation.oracle, oracle: test.oracle, predicted: test.predicted, shuffled: test.shuffled, coefficient: test.coefficient, validationPredicted: validation.predicted, validationShuffled: validation.shuffled },
      sums: { train, validation, test }, testScores,
      metrics: { oracle: test.oracle[k], predicted: test.predicted[k], shuffled: test.shuffled[k], coefficient: test.coefficient[k], baselineSSE: test.baselineSSE, oracleSSE: test.oracleSSE[k],
        predictedSSE: test.predictedSSE ? test.predictedSSE[k] : null, coefficientSSE: test.coefficientSSE ? test.coefficientSSE[k] : null, shuffledSSE: test.shuffledSSE ? test.shuffledSSE[k] : null,
        decompositionGap: context.available ? testScores.reduce((s, r) => s + r.decompositionGap, 0) : null, trainEnergy: rawModel.total,
        numericalRank: rawModel.total > 1e-24 ? rawModel.evals.filter(v => v > 1e-12 * rawModel.total).length : 0, participationRatio: rawModel.total > 1e-24 ? Core.participationRatio(rawModel) : null,
        dimensions: D * length, contextAvailable: context.available },
      notes: ['Random demonstration holdout within task; not an official LIBERO benchmark split.', 'Linear time resampling uses complete demonstration length: offline representation experiment.',
        'Oracle reconstruction uses heldout target trajectories and is not a policy.', 'Initial context is a limited environment proxy; instruction is constant within one task.',
        'Low reconstruction error does not establish dynamics feasibility or task success.'] };
  }
  const api = { unpack, splitIndices, signalInfo, signalRow, analyze, decode, project };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.SubspaceAtlasCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
