/* Bind the atlas' training-space coordinates to a static, world-frame scene.
 * This layer only visualizes signals; it never runs a robot policy or changes samples.
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const Core = window.SubspaceAtlasCore, Path = window.AtlasPath;
  const validPoint = p => p && p.length === 3 && Array.from(p).every(Number.isFinite);
  const points = a => Array.from({length:a.length / 3}, (_, i) => Array.from(a.slice(i * 3, i * 3 + 3)));
  function create(S) {
    let result = null, mapping = null, mappingError = '', displayedScene, lastPayload = null;
    let animation = 0, playing = false, lastTime = 0, progress = 1, frameBounds = null, frameKey = '';
    const view = window.AtlasSceneView.create($('path'), {
      onSelectCandidate: id => window.__atlas.selectCandidate(Number(id)),
      onProgress: p => { stop(); setProgress(p); }
    });
    function stop() { playing = false; cancelAnimationFrame(animation); $('playScene').textContent = '播放轨迹'; }
    function setProgress(p) {
      progress = Math.max(0, Math.min(1, p));
      $('sceneProgress').value = progress * 100;
      $('sceneProgressValue').textContent = Math.round(progress * 100) + '%';
      view.update({progress});
    }
    function tick(now) {
      if (!playing) return;
      setProgress(progress + (now - lastTime) / 5000); lastTime = now;
      if (progress >= 1) stop(); else animation = requestAnimationFrame(tick);
    }
    function options() {
      return {showObjects:$('showSceneObjects').checked, showLabels:$('showSceneLabels').checked,
        showAllCandidates:$('showAllPaths').checked, objectOpacity:Number($('sceneObjectOpacity').value),
        lineWidth:Number($('sceneLineWidth').value), trajectoryOpacity:1,
        coordinateLabel:'世界坐标 · m', frameKey, frameBounds};
    }
    function chooseScene() {
      // One scene is shared by the task's entire training distribution. An
      // observation changes sampling weights, never this spatial reference.
      return S.sceneData?.tasks?.[S.r.groupName] || null;
    }
    function draw() {
      const r = S.r;
      if (!r) return;
      if (result !== r) {
        result = r; stop(); setProgress(1); mapping = null; mappingError = ''; frameKey = '';
        if (!['eepos', 'relative_eepos'].includes(r.signal)) {
          try { mapping = Path.fit(r); } catch (error) { mappingError = error.message; }
        }
      }
      const scene = chooseScene(), query = $('querySource').value === 'test' && S.testQuery?.task === r.groupName ? S.testQuery : null;
      const initial = r.episodes[S.selected]?.data?.eepos?.slice(0, 3);
      const anchor = validPoint(query?.eepos) ? Array.from(query.eepos) : validPoint(initial) ? Array.from(initial) : null;
      const anchorName = validPoint(query?.eepos) ? '当前测试观察' : '所选训练参考';
      $('anchorScene').disabled = !anchor;
      const translate = $('anchorScene').checked && !!anchor;
      const relativeOrigin = translate ? anchor : initial;
      let method, note, diagnostics = null, paths = [];
      let decode;
      if (r.signal === 'eepos') {
        method = 'world-eef-pca'; note = '直接显示末端位置子空间；黑线为训练记录，其他线为线性合成。';
        decode = z => Core.decode(r, z);
      } else if (r.signal === 'relative_eepos' && validPoint(relativeOrigin)) {
        method = 'relative-eef-pca-with-explicit-origin';
        note = '相对末端轨迹以' + (translate ? anchorName : '所选训练参考') + '的初始 EEF 为世界坐标原点。';
        decode = z => Path.toWorld(Core.decode(r, z), relativeOrigin);
      } else if (mapping) {
        method = mapping.kind; diagnostics = Path.diagnostics(mapping, S.k);
        note = '全部 ' + r.episodes.length + ' 条训练示范配对拟合动作坐标 → EEF；当前 k 的训练 3D RMSE ' + (diagnostics.rmse3DMeters * 100).toFixed(2) + ' cm。此误差不代表测试精度。';
        decode = z => Path.decode(mapping, z);
      } else {
        method = 'unavailable'; note = '缺少配对的世界坐标 EEF，暂不叠加轨迹。';
      }
      if (decode) {
        const add = (id, label, flat, kind, color, width, opacity, selected = false) => {
          const p = points(flat);
          if (translate && p.length) {
            const delta = anchor.map((v, d) => v - p[0][d]);
            for (const point of p) for (let d = 0; d < 3; d++) point[d] += delta[d];
          }
          paths.push({id, label, points:p, kind, color, width, opacity, selected});
        };
        let reference;
        try { reference = mapping ? mapping.rows[S.selected] : Core.signalRow(r.episodes[S.selected], r.fields, 'eepos', r.length); } catch (_) { /* Custom data may have no paired EEF. */ }
        if (reference) add('reference', '训练参考 #' + r.episodes[S.selected].id, reference, 'reference', '#34483f', 1.2, .55);
        add('mean', '共同成分', decode([]), 'mean', '#8e9a85', 2.2, 1);
        for (const [i, c] of (S.samples?.candidates || []).entries()) {
          const selected = i === S.candidate;
          add(i, '候选 #' + String(i + 1).padStart(2, '0') + ' · 尚未执行', decode(c.coords.slice(0, S.k)), 'candidate', selected ? '#865da9' : '#a78fbd', selected ? 2.5 : .9, selected ? 1 : .28, selected);
        }
        add('manual', '共同成分 + 手动变化', decode(S.manual.slice(0, S.k)), 'manual', '#167866', 2.3, 1);
      }
      const key = (scene?.key || S.ds.name + '/' + r.groupName) + '/' + r.signal;
      if (key !== frameKey) {
        frameKey = key;
        // Include all true training paths in a fixed frame. Editing one candidate
        // must not resize the scene and disguise the size of that edit.
        let all = mapping ? mapping.rows.flatMap(points) : [];
        if (!mapping && decode) {
          for (const ep of r.episodes) {
            try { all.push(...points(Core.signalRow(ep, r.fields, 'eepos', r.length))); } catch (_) { /* optional */ }
          }
        }
        // Frame the manipulation area, not the full table/floor patch. Large
        // support surfaces remain visible but should not shrink the paths.
        for (const object of scene?.objects || []) {
          if (object.role === 'table') continue;
          const r = object.rotation || [1,0,0,0,1,0,0,0,1];
          for (const x of [-.5,.5]) for (const y of [-.5,.5]) for (const z of [-.5,.5]) {
            const v = [x,y,z].map((q,d) => q * object.size[d]);
            all.push([0,1,2].map(d => object.center[d] + r[d*3]*v[0] + r[d*3+1]*v[1] + r[d*3+2]*v[2]));
          }
        }
        if (!all.length && scene?.bounds) all.push(scene.bounds.min, scene.bounds.max);
        if (validPoint(scene?.eefStart)) all.push(scene.eefStart);
        if (all.length) frameBounds = {min:[0,1,2].map(d => Math.min(...all.map(p => p[d]))), max:[0,1,2].map(d => Math.max(...all.map(p => p[d])))};
        else frameBounds = null;
      }
      $('sceneSourceTag').textContent = scene ? '任务统一场景' : '无场景几何';
      $('sceneSourceTag').classList.toggle('green', !!scene);
      $('sceneSourceNote').textContent = scene
        ? '当前任务的训练全集共用这一布局，用于观察共同成分、变化模式与候选轨迹。切换示范或测试观察时，场景保持不变。'
        : '当前数据没有匹配的场景文件，仅显示可用的世界坐标轨迹。';
      $('pathDescription').textContent = mapping ? '训练配对映射的 EEF 估计 · 非仿真执行' : decode ? '末端位置子空间 · 世界坐标（米）' : '场景空间参照 · EEF 映射不可用';
      $('scenePathNote').textContent = note + (translate ? ' 已将各条轨迹起点平移至' + anchorName + '；仅改变显示，不改变动作。' : '') + ' 物体保持初态，播放只展示轨迹进度。';
      lastPayload = {sceneKey:scene?.key || null, sceneScope:'task-training-ensemble', sceneProvenance:scene?.provenance || null, frame:'world', units:'meters', method, diagnostics,
        mappingError:mappingError || null, alignment:{enabled:translate, target:translate ? anchor : null, source:translate ? anchorName : null, displayOnly:true},
        relativeOrigin:r.signal === 'relative_eepos' ? Array.from(relativeOrigin || []) : null, paths};
      const payload = {paths, progress, options:options()};
      if (scene !== displayedScene) { payload.scene = scene; displayedScene = scene; }
      view.update(payload);
    }
    for (const id of ['showSceneObjects', 'showSceneLabels', 'showAllPaths']) $(id).addEventListener('change', () => view.update({options:options()}));
    for (const id of ['sceneObjectOpacity', 'sceneLineWidth']) $(id).addEventListener('input', () => view.update({options:options()}));
    $('anchorScene').addEventListener('change', draw);
    $('sceneCamera').addEventListener('change', () => view.setView($('sceneCamera').value));
    $('resetView').addEventListener('click', () => { view.reset(); $('sceneCamera').value = 'perspective'; });
    $('sceneProgress').addEventListener('input', () => { stop(); setProgress(Number($('sceneProgress').value) / 100); });
    $('playScene').addEventListener('click', () => {
      if (playing) { stop(); return; }
      if (progress >= 1) setProgress(0);
      playing = true; $('playScene').textContent = '暂停'; lastTime = performance.now(); animation = requestAnimationFrame(tick);
    });
    function expand(on) {
      $('sceneCard').classList.toggle('scene-expanded', on);
      $('expandScene').textContent = on ? '缩小场景' : '放大场景';
      $('expandScene').setAttribute('aria-expanded', String(on));
      requestAnimationFrame(() => view.update({}));
    }
    $('expandScene').addEventListener('click', () => expand(!$('sceneCard').classList.contains('scene-expanded')));
    document.addEventListener('keydown', e => { if (e.key === 'Escape') expand(false); });
    function clear() {
      stop(); result = null; displayedScene = null; lastPayload = null; frameKey = ''; frameBounds = null;
      view.update({scene:null, paths:[], progress:1, options:{frameKey:'empty', frameBounds:null}});
      $('sceneSourceTag').textContent = '暂无场景';
      $('sceneSourceNote').textContent = '请载入可分析的训练数据。';
      $('scenePathNote').textContent = '';
    }
    return {draw, clear, view, exportState:() => lastPayload ? {...lastPayload, progress, camera:view.getState(), visibility:options(),
      notes:'Static initialization geometry and displayed EEF paths only. No policy rollout, object motion, collision checking or success evaluation. Display alignment never changes exported action candidates.'} : null};
  }
  window.AtlasSceneBridge = {create};
})();
