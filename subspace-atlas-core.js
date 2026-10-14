/* Subspace Atlas: every supplied training demonstration defines the task space.
 * No data split or test action target. PCA rank describes training variance.
 * Sampling uses only initial observations and the training coefficient mixture.
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
  function channelScale(rows, D) {
    const shift = new Float64Array(D), scale = new Float64Array(D), count = rows.length * rows[0].length / D, constantChannels = [];
    rows.forEach(row => { for (let j = 0; j < row.length; j++) shift[j % D] += row[j] / count; });
    rows.forEach(row => { for (let j = 0; j < row.length; j++) scale[j % D] += (row[j] - shift[j % D]) ** 2 / count; });
    for (let d = 0; d < D; d++) { scale[d] = Math.sqrt(scale[d]); if (!finite(scale[d]) || !finite(shift[d])) fail('Numeric range too large for stable scaling'); if (scale[d] <= 1e-12 * Math.max(1, Math.abs(shift[d]))) { constantChannels.push(d); scale[d] = 1; } }
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
function analyze(input,opts){
 if(!Core)fail('Load ActionCore before SubspaceAtlasCore');
 const ds=unpack(input);opts=opts||{};
 let group=opts.group==null?0:opts.group;
 if(typeof group==='string')group=ds.groups.indexOf(group);
 if(!integer(group,0)||group>=ds.groups.length)fail('Select exactly one valid task group');
 const length=opts.length==null?64:opts.length,signal=opts.signal||'action';
 if(!integer(length,2)||length>1024)fail('Length must be 2..1024');
 const sourceIndices=ds.episodes.map((_,i)=>i).filter(i=>ds.episodes[i].group===group);
 if(sourceIndices.length<2)fail('At least two training demonstrations are required for the selected task');
 const episodes=sourceIndices.map(i=>ds.episodes[i]),info=signalInfo(ds.fields,signal),D=info.D;
 const trainingIndices=episodes.map((_,i)=>i),rows=episodes.map(ep=>signalRow(ep,ds.fields,signal,length)),scaling=channelScale(rows,D);
 const normalized=rows.map(row=>Float64Array.from(row,(v,j)=>(v-scaling.shift[j%D])/scaling.scale[j%D]));
 const rankLimit=Math.min(episodes.length-1,D*length),rawModel=Core.pca(normalized,rankLimit),comps=[],eigenvalues=[];
 rawModel.comps.forEach((candidate,k)=>{
  if(!(rawModel.total>1e-24)||!(rawModel.evals[k]>1e-12*rawModel.total))return;
  const c=Float64Array.from(candidate);
  for(let pass=0;pass<2;pass++)for(const prev of comps){const dot=Core.dot(prev,c);for(let j=0;j<c.length;j++)c[j]-=dot*prev[j];}
  const norm=Math.sqrt(Core.dot(c,c));if(norm<1e-10)return;
  for(let j=0;j<c.length;j++)c[j]/=norm;
  let largest=0;for(let j=1;j<c.length;j++)if(Math.abs(c[j])>Math.abs(c[largest]))largest=j;
  if(c[largest]<0)for(let j=0;j<c.length;j++)c[j]*=-1;
  comps.push(c);eigenvalues.push(rawModel.evals[k]);
 });
 const model={mean:rawModel.mean,comps,eigenvalues:Float64Array.from(eigenvalues),evals:Float64Array.from(eigenvalues),
  sd:Float64Array.from(eigenvalues,v=>Math.sqrt(v/Math.max(1,episodes.length-1))),shift:scaling.shift,scale:scaling.scale,
  constantChannels:scaling.constantChannels,total:rawModel.total,n:episodes.length,exact:rawModel.exact};
 const coords=normalized.map(row=>project(row,model)),evr=eigenvalues.map(v=>v/rawModel.total);let running=0;
 const cumulative=[rawModel.total>1e-24?0:null,...evr.map(v=>Math.min(1,running+=v))];
 const curveResidual=cumulative.map(v=>v==null?null:Math.max(0,1-v));
 const thresholdIndex=cumulative.findIndex(v=>v!=null&&v>=.9-1e-12),selectedRank=thresholdIndex<0?comps.length:thresholdIndex;
 return{name:ds.name,group,groupName:ds.groups[group],signal,signalInfo:info,labels:info.labels,D,length,episodes,sourceIndices,
  fields:ds.fields,contextInfo:ds.contextInfo,rows,normalized,trainingIndices,model,coords,maxRank:comps.length,rankLimit,
  evr,cumulative,curveResidual,selectedRank,rankTarget:.9,
  selectionReason:rawModel.total>1e-24?'Smallest rank explaining at least 90% of all supplied training variance (descriptive)':'No across-demonstration training variance; relative variance is undefined',
  notes:['Every demonstration in the selected task fits the space; none is held out.',
   'Rank and residual describe training representation only; no generalization or success score is computed.',
   'Full trajectories are linearly resampled in time; no action-aware alignment or DTW is used.',
   'Sampling returns candidate trajectories, not verified feasible robot policies.',
   'Conditional queries contain initial observations only; no query action trajectory is required.']};
}
function sample(result,opts){
 opts=opts||{};
 const count=opts.count==null?24:opts.count,rank=opts.rank==null?result.selectedRank:opts.rank;
 const seed=opts.seed==null?42:opts.seed,temperature=opts.temperature==null?.2:opts.temperature;
 const bandwidth=opts.bandwidth==null?1:opts.bandwidth,mode=opts.contextMode||'none',query=opts.query==null?null:opts.query;
 if(!integer(count,1)||count>10000)fail('Candidate count must be 1..10000');
 if(!integer(rank,0)||rank>result.maxRank)fail('Rank must lie within the fitted training space');
 if(!integer(seed,0)||seed>4294967295)fail('Seed must be uint32');
 if(!finite(temperature)||temperature<0)fail('Temperature must be finite and nonnegative');
 if(!finite(bandwidth)||bandwidth<=0)fail('Bandwidth must be finite and positive');
 if(!['none','eepos','observation'].includes(mode))fail('Unknown context mode');
 if(query!=null&&(typeof query!=='object'||Array.isArray(query)||Object.keys(query).some(k=>!['eepos','context'].includes(k))))fail('Query accepts only initial eepos/context observations; action or unknown fields are forbidden');
 if(mode==='none'&&query!=null)fail('Unconditional sampling requires query:null; choose a context mode for observations');
 if(mode!=='none'&&query==null)fail('Conditional sampling requires a compatible initial observation query');
 const N=result.episodes.length,weights=new Float64Array(N).fill(1/N),distances=new Float64Array(N),warnings=[];
 let informativeDimensions=0,nearestDistance=null,contextScale=null;
 if(mode!=='none'){
  let raw,q;
  if(mode==='eepos'){
   const f=result.fields.find(f=>f.name==='eepos');
   if(!f||f.dim<3||result.episodes.some(ep=>!ep.data.eepos||ep.data.eepos.length<3))fail('Training demonstrations lack initial eepos XYZ');
   raw=result.episodes.map(ep=>ep.data.eepos.slice(0,3));q=vector(query.eepos,3,'Query initial eepos');
  }else{
   const P=result.episodes[0].context&&result.episodes[0].context.length;
   if(!P||result.episodes.some(ep=>!ep.context||ep.context.length!==P))fail('Training demonstrations lack compatible initial observation context vectors');
   raw=result.episodes.map(ep=>ep.context);q=vector(query.context,P,'Query initial context');
  }
  contextScale=channelScale(raw,q.length);
  const inactive=new Set(contextScale.constantChannels),active=Array.from({length:q.length},(_,i)=>i).filter(i=>!inactive.has(i));
  informativeDimensions=active.length;
  if(!active.length)warnings.push('All training context dimensions are constant; no observation-based weighting is possible, so weights are uniform.');
  else{
   for(let i=0;i<N;i++){let squared=0;for(const d of active)squared+=((raw[i][d]-q[d])/contextScale.scale[d])**2;distances[i]=squared/active.length;}
   let nearestSquared=Infinity;for(const d of distances)nearestSquared=Math.min(nearestSquared,d);
   if(!finite(nearestSquared))fail('Query numeric range is too large for stable context distances');
   nearestDistance=Math.sqrt(nearestSquared);let mass=0;
   for(let i=0;i<N;i++){
    const delta=distances[i]-nearestSquared;
    weights[i]=delta===0?1:Math.exp(-.5*(Math.sqrt(delta)/bandwidth)**2);mass+=weights[i];
   }
   for(let i=0;i<N;i++)weights[i]/=mass;
  }
 }
 const rng=Core.makeRng(seed),candidates=[];
 for(let c=0;c<count;c++){
  const u=rng();let chosen=N-1,mass=0;
  for(let i=0;i<N;i++){mass+=weights[i];if(u<mass){chosen=i;break;}}
  const coords=Float64Array.from(result.coords[chosen].slice(0,rank),(v,k)=>v+temperature*result.model.sd[k]*rng.normal());
  if(coords.some(v=>!finite(v)))fail('Temperature produces non-finite sampled coordinates');
  candidates.push({coords,sourceIndex:result.sourceIndices[chosen],index:chosen});
 }
 return{candidates,weights,effectiveNeighbors:1/weights.reduce((sum,w)=>sum+w*w,0),nearestDistance,
  distances,distanceKind:'mean squared standardized distance (nearestDistance is its square root)',mode,informativeDimensions,contextScale,warnings,rank,seed,temperature,bandwidth};
}
const api={unpack,signalInfo,signalRow,analyze,sample,decode,project};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
root.SubspaceAtlasCore=api;
})(typeof globalThis!=='undefined'?globalThis:this);
