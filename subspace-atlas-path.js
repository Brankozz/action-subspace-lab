/* Map trajectory PCA coordinates to paired, recorded world-frame EEF paths.
 * All fitting data are the selected task's training demonstrations. This is a
 * descriptive linear association, never a controller rollout or feasibility test.
 */
(function (root) {
  'use strict';
  const Numeric = typeof module !== 'undefined' && module.exports ? require('./atlas-numeric-core.js') : root.ActionCore;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const arrayLike = value => Array.isArray(value) || ArrayBuffer.isView(value);
  function fail(message) { throw new Error(message); }
  function numericVector(value, length, label) {
    if (!arrayLike(value) || value.length !== length) fail(label + ': unexpected vector length');
    for (const x of value) if (!finite(x)) fail(label + ': non-finite value');
  }

  function fit(result) {
    if (!Numeric) fail('Load ActionCore before AtlasPath');
    if (!result || !Array.isArray(result.episodes) || result.episodes.length < 2) fail('At least two paired training demonstrations are required');
    const field = result.fields && result.fields.find(f => f.name === 'eepos');
    if (!field || !Number.isInteger(field.dim) || field.dim < 3) fail('Paired training data do not contain world EEF XYZ');
    const length = result.length, rank = result.model && result.model.comps.length;
    if (!Number.isInteger(length) || length < 2 || length > 1024 || !Number.isInteger(rank) || rank < 0) fail('Invalid fitted trajectory model');
    const n = result.episodes.length, size = 3 * length;
    if (!Array.isArray(result.coords) || result.coords.length !== n) fail('Every training demonstration must have matching PCA coordinates');
    const coordinates = result.coords.map((z, i) => {
      numericVector(z, rank, 'Training coordinate ' + i);
      return Float64Array.from(z);
    });
    const rows = result.episodes.map((ep, i) => {
      if (!Number.isInteger(ep.T) || ep.T < 1) fail('Invalid training length ' + i);
      const values = ep.data && ep.data.eepos;
      numericVector(values, ep.T * field.dim, 'Training EEF path ' + i);
      const xyz = new Float64Array(ep.T * 3);
      for (let t = 0; t < ep.T; t++) for (let d = 0; d < 3; d++) xyz[t * 3 + d] = values[t * field.dim + d];
      return Numeric.resample(xyz, ep.T, 3, length);
    });
    const mean = new Float64Array(size), scoreMean = new Float64Array(rank), scoreScale = new Float64Array(rank);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < size; j++) mean[j] += rows[i][j] / n;
      for (let k = 0; k < rank; k++) scoreMean[k] += coordinates[i][k] / n;
    }
    for (let i = 0; i < n; i++) for (let k = 0; k < rank; k++) scoreScale[k] += (coordinates[i][k] - scoreMean[k]) ** 2 / n;
    for (let k = 0; k < rank; k++) scoreScale[k] = Math.sqrt(scoreScale[k]);
    const maxScale = Math.max(0, ...scoreScale), active = [];
    for (let k = 0; k < rank; k++) {
      // Core PCA removes numerical zero eigenvalues. Keep this guard for constant
      // coordinates and externally constructed result objects as well.
      if (scoreScale[k] > Math.max(1e-14, maxScale * 1e-12)) active.push(k);
      if (Math.abs(scoreMean[k]) > Math.max(1e-10, scoreScale[k] * 1e-7)) fail('EEF mapping requires centered PCA coordinates');
    }
    const basis = Array.from({ length: rank }, () => new Float64Array(size)), p = active.length;
    let effectiveRank = 0, maxScoreCorrelation = 0;
    if (p) {
      // Standardize internally only to condition the solve. The returned basis
      // multiplies the original, unwhitened PCA coordinates, not z / model.sd.
      const normalized = coordinates.map(z => Float64Array.from(active, k => (z[k] - scoreMean[k]) / scoreScale[k]));
      const gram = new Float64Array(p * p), cross = Array.from({ length: p }, () => new Float64Array(size));
      for (let i = 0; i < n; i++) for (let a = 0; a < p; a++) {
        const za = normalized[i][a] / n;
        for (let b = 0; b < p; b++) gram[a * p + b] += za * normalized[i][b];
        for (let j = 0; j < size; j++) cross[a][j] += za * (rows[i][j] - mean[j]);
      }
      for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) maxScoreCorrelation = Math.max(maxScoreCorrelation, Math.abs(gram[a * p + b]));
      const eigen = Numeric.jacobi(gram, p), threshold = Math.max(1e-12, eigen.values[0] * 1e-10);
      for (let q = 0; q < p; q++) {
        if (eigen.values[q] <= threshold) continue;
        effectiveRank++;
        const v = eigen.vectors[q], projected = new Float64Array(size);
        for (let a = 0; a < p; a++) for (let j = 0; j < size; j++) projected[j] += v[a] * cross[a][j] / eigen.values[q];
        for (let a = 0; a < p; a++) {
          const k = active[a], weight = v[a] / scoreScale[k];
          for (let j = 0; j < size; j++) basis[k][j] += weight * projected[j];
        }
      }
    }
    const sseByRank = new Float64Array(rank + 1);
    for (let i = 0; i < n; i++) {
      const prediction = Float64Array.from(mean);
      for (let k = 0; k <= rank; k++) {
        if (k) for (let j = 0; j < size; j++) prediction[j] += basis[k - 1][j] * coordinates[i][k - 1];
        for (let j = 0; j < size; j++) sseByRank[k] += (prediction[j] - rows[i][j]) ** 2;
      }
    }
    if (mean.some(v => !finite(v)) || basis.some(row => row.some(v => !finite(v))) || sseByRank.some(v => !finite(v))) fail('Numeric range too large for EEF fitting');
    return {
      kind: 'paired-training-linear-eef-estimate', version: 1, frame: 'world', units: 'meters',
      signal: result.signal, length, rank, trainingCount: n,
      trainingEpisodeIds: result.episodes.map(ep => ep.id),
      mean, basis, rows, scoreMean, scoreScale, effectiveRank, maxScoreCorrelation,
      diagnostics: { scope: 'all-training-demonstrations-only', sseByRank, baselineSSE: sseByRank[0] },
      notes: [
        'Every selected-task training demonstration contributes paired PCA coordinates and recorded EEF positions.',
        'World EEF positions and source signals use the same linear normalized-time resampling.',
        'The basis multiplies raw, unwhitened PCA coordinates: EEF(z) = meanEEF + basis * z.',
        'This association does not simulate a controller, contact, grasping, object motion, or a test trajectory.',
        'It does not condition the EEF mapping on a query scene or enforce a query initial position.',
        'A low training fit error, including an exact high-rank fit, is not evidence of generalization or task success.'
      ]
    };
  }

  function decode(model, coordinates) {
    if (!model || model.kind !== 'paired-training-linear-eef-estimate') fail('Expected paired training EEF model');
    if (!arrayLike(coordinates) || coordinates.length > model.rank) fail('EEF coordinates exceed fitted rank');
    numericVector(coordinates, coordinates.length, 'EEF coordinates');
    const path = Float64Array.from(model.mean);
    for (let k = 0; k < coordinates.length; k++) for (let j = 0; j < path.length; j++) path[j] += coordinates[k] * model.basis[k][j];
    if (path.some(v => !finite(v))) fail('EEF coordinates produce a non-finite path');
    return path;
  }

  function diagnostics(model, rank) {
    if (rank == null) rank = model.rank;
    if (!Number.isInteger(rank) || rank < 0 || rank > model.rank) fail('Invalid EEF diagnostic rank');
    const sse = model.diagnostics.sseByRank[rank], baseline = model.diagnostics.baselineSSE;
    return {
      scope: 'all-training-demonstrations-only', rank, trainingCount: model.trainingCount, sse,
      rmse3DMeters: Math.sqrt(sse / (model.trainingCount * model.length)),
      relativeSSE: baseline > 1e-20 ? sse / baseline : null
    };
  }

  function toWorld(relativePath, initialEef) {
    if (!arrayLike(relativePath) || relativePath.length < 3 || relativePath.length % 3) fail('Expected flattened relative EEF XYZ path');
    numericVector(relativePath, relativePath.length, 'Relative EEF path');
    numericVector(initialEef, 3, 'Known initial EEF position');
    const path = Float64Array.from(relativePath, (v, j) => v + initialEef[j % 3]);
    if (path.some(v => !finite(v))) fail('Initial EEF position produces a non-finite path');
    return path;
  }

  const api = { fit, decode, diagnostics, toWorld };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.AtlasPath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
