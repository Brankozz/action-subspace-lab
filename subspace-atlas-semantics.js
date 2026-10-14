/* Exploratory task-mean / instruction-word association. No DOM and no network.
 * The two matrices use the same task order; undefined cosine/correlation is null.
 * Whole task labels are permuted, never individual (dependent) task pairs. */
(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./subspace-atlas-core.js'));
  } else {
    root.AtlasSemantics = factory(root.SubspaceAtlasCore);
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function (AtlasCore) {
  'use strict';

  const MIN_DEMOS = 2;
  const PERMUTATIONS = 199;
  // Keep spatial/directional words (in, on, under, left, right, to, from, ...).
  const STOPWORDS = new Set(('a an the and then please is are was were be been being ' +
    'this that these those it its of').split(' '));

  function tokens(text) {
    return (String(text).toLowerCase().match(/[a-z]+/g) || [])
      .filter(function (word) { return word.length > 1 && !STOPWORDS.has(word); });
  }

  function textVectors(labels) {
    const counts = labels.map(function (label) {
      const count = new Map();
      for (const word of tokens(label)) count.set(word, (count.get(word) || 0) + 1);
      return count;
    });
    const documentFrequency = new Map();
    for (const count of counts) {
      for (const word of count.keys()) {
        documentFrequency.set(word, (documentFrequency.get(word) || 0) + 1);
      }
    }
    const vocabulary = Array.from(documentFrequency.keys()).sort();
    return {
      vocabulary: vocabulary,
      vectors: counts.map(function (count) {
        return Float64Array.from(vocabulary, function (word) {
          const frequency = count.get(word) || 0;
          const idf = 1 + Math.log((1 + labels.length) / (1 + documentFrequency.get(word)));
          return frequency > 0 ? (1 + Math.log(frequency)) * idf : 0;
        });
      })
    };
  }

  function suppliedTextVectors(labels, vectors) {
    let dimension = null;
    return labels.map(function (label) {
      const row = Object.prototype.hasOwnProperty.call(vectors, label) ? vectors[label] : null;
      if ((!Array.isArray(row) && !ArrayBuffer.isView(row)) || !row.length) {
        throw new Error('Missing contextual text vector for task: ' + label);
      }
      if (dimension === null) dimension = row.length;
      if (row.length !== dimension || !Array.from(row).every(Number.isFinite)) {
        throw new Error('Contextual text vectors must be finite and share a dimension.');
      }
      return Float64Array.from(row);
    });
  }

  function cosineMatrix(vectors) {
    const normalized = vectors.map(function (v) {
      let square = 0;
      for (const value of v) square += value * value;
      const norm = Math.sqrt(square);
      if (!(norm > 1e-12)) return null;
      return Float64Array.from(v, function (value) { return value / norm; });
    });
    return normalized.map(function (a, i) {
      return normalized.map(function (b, j) {
        if (!a || !b) return null;
        if (i === j) return 1;
        let sum = 0;
        for (let k = 0; k < a.length; k++) sum += a[k] * b[k];
        return Math.max(-1, Math.min(1, sum));
      });
    });
  }

  function pearson(a, b) {
    if (a.length !== b.length || a.length < 2) return null;
    let am = 0, bm = 0;
    for (let i = 0; i < a.length; i++) { am += a[i]; bm += b[i]; }
    am /= a.length; bm /= b.length;
    let aa = 0, bb = 0, ab = 0;
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - am, db = b[i] - bm;
      aa += da * da; bb += db * db; ab += da * db;
    }
    // Cosine values are bounded; this catches numerical noise in constant arrays.
    if (aa <= a.length * 1e-24 || bb <= b.length * 1e-24) return null;
    return Math.max(-1, Math.min(1, ab / Math.sqrt(aa * bb)));
  }

  function rng(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state + 0x6d2b79f5) | 0;
      let value = Math.imul(state ^ (state >>> 15), 1 | state);
      value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function association(action, text, seed) {
    // Restrict to complete task nodes, so every permutation has the same pairs.
    const taskIndices = action.map(function (_, i) { return i; }).filter(function (i) {
      return action[i][i] !== null && text[i][i] !== null;
    });
    const a = [], b = [];
    for (let i = 0; i < taskIndices.length; i++) {
      for (let j = i + 1; j < taskIndices.length; j++) {
        a.push(action[taskIndices[i]][taskIndices[j]]);
        b.push(text[taskIndices[i]][taskIndices[j]]);
      }
    }
    const correlation = pearson(a, b);
    let pValue = null;
    if (correlation !== null && taskIndices.length >= 3) {
      const random = rng((seed >>> 0) ^ 0x53544d);
      let extreme = 0;
      for (let permutation = 0; permutation < PERMUTATIONS; permutation++) {
        const order = taskIndices.slice();
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(random() * (i + 1));
          const temp = order[i]; order[i] = order[j]; order[j] = temp;
        }
        const permuted = [];
        for (let i = 0; i < order.length; i++) {
          for (let j = i + 1; j < order.length; j++) permuted.push(text[order[i]][order[j]]);
        }
        const value = pearson(a, permuted);
        if (value !== null && Math.abs(value) >= Math.abs(correlation) - 1e-12) extreme++;
      }
      pValue = (extreme + 1) / (PERMUTATIONS + 1);
    }
    return { correlation: correlation, pValue: pValue, nPairs: a.length, taskIndices: taskIndices };
  }

  function analyze(ds, options) {
    options = options || {};
    if (!AtlasCore || !AtlasCore.signalRow || !AtlasCore.signalInfo) {
      throw new Error('AtlasSemantics requires subspace-atlas-core.js.');
    }
    if (!ds || !Array.isArray(ds.groups) || !Array.isArray(ds.episodes) || !Array.isArray(ds.fields)) {
      throw new Error('Expected a dataset with groups, episodes, and fields.');
    }
    const signal = options.signal || 'action';
    const length = options.length == null ? 64 : Number(options.length);
    const seed = options.seed == null ? 42 : Number(options.seed);
    if (!Number.isInteger(length) || length < 2) throw new Error('length must be an integer >= 2.');
    if (!Number.isFinite(seed)) throw new Error('seed must be finite.');
    const info = AtlasCore.signalInfo(ds.fields, signal);
    const D = info.D;
    const groups = ds.groups.map(function () { return []; });
    ds.episodes.forEach(function (episode, index) {
      if (Number.isInteger(episode.group) && groups[episode.group]) groups[episode.group].push(index);
    });
    const labels = [], groupIndices = [], means = [], trainingIndices = [], skippedGroups = [];
    const channelMean = new Float64Array(D), channelM2 = new Float64Array(D);
    let observations = 0;
    groups.forEach(function (indices, group) {
      if (indices.length < MIN_DEMOS) {
        skippedGroups.push({ group: group, label: String(ds.groups[group]), count: indices.length,
          reason: 'fewer_than_2_demonstrations' });
        return;
      }
      const train = indices.slice();
      const mean = new Float64Array(length * D);
      train.forEach(function (index, count) {
        const row = AtlasCore.signalRow(ds.episodes[index], ds.fields, signal, length);
        if (row.length !== mean.length) throw new Error('Inconsistent signal dimensions.');
        for (let t = 0; t < length; t++) {
          observations++;
          for (let d = 0; d < D; d++) {
            const k = t * D + d, value = row[k];
            if (!Number.isFinite(value)) throw new Error('Training signal contains non-finite values.');
            mean[k] += (value - mean[k]) / (count + 1);
            const delta = value - channelMean[d];
            channelMean[d] += delta / observations;
            channelM2[d] += delta * (value - channelMean[d]);
          }
        }
      });
      labels.push(String(ds.groups[group])); groupIndices.push(group);
      means.push(mean); trainingIndices.push(train);
    });
    const channelScale = Float64Array.from(channelM2, function (m2) {
      const std = Math.sqrt(Math.max(0, m2 / Math.max(1, observations)));
      return std > 1e-12 ? std : 1;
    });
    const grandMean = new Float64Array(length * D);
    means.forEach(function (mean) {
      for (let k = 0; k < grandMean.length; k++) grandMean[k] += mean[k] / means.length;
    });
    const centered = means.map(function (mean) {
      return Float64Array.from(mean, function (value, k) {
        return (value - grandMean[k]) / channelScale[k % D];
      });
    });
    const contextual = options.textVectors != null;
    const text = contextual
      ? { vocabulary: [], vectors: suppliedTextVectors(labels, options.textVectors) }
      : textVectors(labels);
    const actionSimilarity = cosineMatrix(centered), textSimilarity = cosineMatrix(text.vectors);
    const result = association(actionSimilarity, textSimilarity, seed);
    const notes = [
      contextual
        ? '探索性语言表征关联：使用外部提供的指令向量计算余弦。页面提供的 Qwen3-4B 向量是纯文本最后层隐藏状态的非特殊、非填充 token 均值；它不是专门校准的句向量，不能证明语义与环境已解耦。'
        : '探索性词汇关联：英文指令经小写分词、停用词过滤和 TF-IDF 后计算余弦；这不是语义嵌入，也不能证明语义与环境已解耦。',
      '每任务至少 2 条演示；每个任务的全部训练示范参与均值和尺度估计，不拆分训练集。种子仅用于任务标签置换。',
      '动作均值使用统一 phase 重采样；所有入选任务的训练轨迹共同估计每通道标准差，再减去等权任务均值的总体均值计算余弦。动作形状相似度可为负。',
      '相关系数取两矩阵上三角。p 值为 199 次整任务标签置换的双侧 Monte Carlo 检验，并用 (极端次数 + 1) / 200 修正；没有把相关任务对当作独立样本。',
      '布局、物体、任务命名和数据来源仍可能混杂；置换检验依赖任务标签可交换假设。多次选择信号或种子后的 p 值不能作为预先指定检验的证据。'
    ];
    if (labels.length < 3) notes.push('可用任务少于 3 个，无法估计任务间关联。');
    if (result.taskIndices.length < labels.length) notes.push('零范数动作均值或空词向量对应的任务不进入相关与置换；矩阵中的未定义余弦显示为 null。');
    if (result.correlation === null) notes.push('任务对不足或相似度为常数，相关系数和 p 值未定义。');
    if (signal === 'cumaction') notes.push('累计动作仅为动作积分示意，不等于已观测或已验证的末端路径。');
    return {
      labels: labels, groupIndices: groupIndices,
      actionSimilarity: actionSimilarity, textSimilarity: textSimilarity,
      correlation: result.correlation, pValue: result.pValue, nPairs: result.nPairs,
      inferenceTaskIndices: result.taskIndices,
      trainingIndices: trainingIndices, trainingCounts: trainingIndices.map(function (train) { return train.length; }),
      skippedGroups: skippedGroups, channelScale: Array.from(channelScale),
      channelLabels: info.labels, signal: signal, length: length, seed: seed,
      permutations: PERMUTATIONS, vocabulary: text.vocabulary,
      textMethod: contextual ? 'contextual' : 'tfidf',
      textMethodLabel: contextual ? 'Contextual instruction-vector cosine' : 'Lexical TF-IDF cosine',
      textDimension: text.vectors.length ? text.vectors[0].length : 0,
      method: 'centered-task-mean cosine vs ' + (contextual ? 'contextual instruction-vector' : 'lexical TF-IDF') + ' cosine; two-sided task-label permutation',
      notes: notes, note: notes.join(' ')
    };
  }

  return { analyze: analyze };
}));
