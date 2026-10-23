#!/usr/bin/env python
"""Build a self-contained HTML page for tuning the GFP background/foreground mask by eye.

Usage:
    python make_tuning_page.py <data_dir> <out.html>

<data_dir> holds P.npy (empty-dish GFP map, 1200x1200 uint16: per-pixel 1st percentile across
raw tiles) and one <tag>.npz per crop with gfp, rfp (uint16 montage crops), y0, x0 (crop origin in
the 3360^2 robo4_serpentine montage, 1080 px stride) and tiles = rows of (tile, i, j, s, sigma):
the per-tile fitted background scale (background = s * P) and background noise.

Mask rule (mirrors what the batch version will do):
    smooth = gaussian(GFP, smoothing)
    background level = level * s_tile * P    ("per-tile fit")   or   level * P   ("fixed")
    foreground = smooth > background level + k * noise
    then drop foreground specks < min_speck px and fill holes < max_hole px (4-connected).
"""
import base64
import glob
import json
import os
import sys

import numpy as np

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Foreground Tuning</title>
<style>
:root{--bg:#15171c;--panel:#20232a;--text:#e6e6e6;--muted:#9aa0aa;--accent:#4c8dff}
body{margin:0;background:var(--bg);color:var(--text);font:13px -apple-system,system-ui,sans-serif}
#top{display:grid;grid-template-columns:330px 1fr;gap:14px;padding:12px 16px}
.card{background:var(--panel);border-radius:8px;padding:12px}
.row{margin:9px 0}.row label{display:flex;justify-content:space-between;color:var(--muted)}
.row input[type=range]{width:100%}
b.v{color:var(--text);font-weight:600}
button{background:#2c313a;color:var(--text);border:1px solid #3a404b;border-radius:6px;padding:5px 9px;cursor:pointer;margin:2px}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;text-align:right;border-bottom:1px solid #2e333c}th{color:var(--muted);font-weight:500}
td:first-child,th:first-child{text-align:left}
tr.cur td{background:#26324a}
#wrap{overflow:auto;max-height:calc(100vh - 40px)}
#stage{position:relative;transform-origin:0 0}#stage canvas{position:absolute;left:0;top:0;image-rendering:pixelated}
pre{background:#111318;padding:8px;border-radius:6px;white-space:pre-wrap;margin:6px 0}
.muted{color:var(--muted)}
</style></head><body>
<div id="top">
<div>
 <div class="card">
  <div class="row"><span class="muted">Background model</span><br>
   <button id="m_tile" class="on">Per-tile fit</button><button id="m_fixed">Fixed empty-dish</button></div>
  <div class="row"><label>Background level × <b class="v" id="v_level"></b></label><input id="level" type="range" min="30" max="300" value="100"></div>
  <div class="row"><label>Threshold above background (noise units) <b class="v" id="v_k"></b></label><input id="k" type="range" min="0" max="300" value="40"></div>
  <div class="row"><label>Smoothing (px) <b class="v" id="v_sm"></b></label><input id="sm" type="range" min="0" max="60" value="15"></div>
  <div class="row"><label>Remove specks smaller than (px) <b class="v" id="v_speck"></b></label><input id="speck" type="range" min="0" max="3000" step="25" value="200"></div>
  <div class="row"><label>Fill holes smaller than (px) <b class="v" id="v_hole"></b></label><input id="hole" type="range" min="0" max="6000" step="50" value="300"></div>
  <div class="row"><button id="reset">Reset defaults</button></div>
 </div>
 <div class="card" style="margin-top:12px">
  <div class="row"><span class="muted">View</span><br>
   <button id="b_gfp" class="on">GFP</button><button id="b_rfp">RFP</button>
   <button id="ov" class="on">Overlay (H)</button><button id="ol" class="on">Outline</button></div>
  <div class="row"><span class="muted">Zoom</span> <button data-z="1" class="zb on">1×</button><button data-z="2" class="zb">2×</button><button data-z="3" class="zb">3×</button></div>
  <div class="row"><label>Image gamma <b class="v" id="v_gm"></b></label><input id="gm" type="range" min="20" max="150" value="60"></div>
  <div class="row"><label>Background tint opacity <b class="v" id="v_op"></b></label><input id="op" type="range" min="0" max="100" value="45"></div>
 </div>
 <div class="card" style="margin-top:12px"><span class="muted">Current settings (send me these)</span>
  <pre id="settings"></pre><button id="copy">Copy</button></div>
</div>
<div>
 <div class="card"><table id="tbl"><thead><tr><th>Crop</th><th>Foreground %</th><th>GFP fg</th><th>RFP fg</th><th>RFP/GFP</th><th>bg-sub RFP/GFP</th><th>GFP bg</th><th>RFP bg</th></tr></thead><tbody></tbody></table>
 <div class="muted" style="margin-top:6px">Click a row to view it. Blue tint = called background; yellow line = foreground edge. bg-sub ratio = (RFP fg − RFP bg) / (GFP fg − GFP bg), bg = median over background pixels.</div></div>
 <div id="wrap" style="margin-top:12px"><div id="stage"><canvas id="cv"></canvas></div></div>
</div></div>
<script>
const DATA = __DATA__;
const N = DATA.crops[0].n, STRIDE = 1080, FIXED_SIGMA = 30;
const dec = b64 => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Uint16Array(u.buffer); };
const P = dec(DATA.P);
const DEF = {mode: 'tile', level: 100, k: 40, sm: 15, speck: 200, hole: 300};
let prm = Object.assign({}, DEF), cur = 0, chan = 'gfp', zoom = 1, showOv = true, showOl = true;
const crops = DATA.crops.map(c => {
  const g = dec(c.gfp), r = dec(c.rfp), n = c.n;
  const Pc = new Float32Array(n * n), tid = new Uint8Array(n * n);
  const S = {}, SG = {}; c.tiles.forEach(t => { S[t[1] * 3 + t[2]] = t[3]; SG[t[1] * 3 + t[2]] = t[4]; });
  for (let y = 0; y < n; y++){ const Y = c.y0 + y, i = Math.min(2, Math.floor(Y / STRIDE)), ly = Y - i * STRIDE;
    for (let x = 0; x < n; x++){ const X = c.x0 + x, j = Math.min(2, Math.floor(X / STRIDE)), lx = X - j * STRIDE;
      Pc[y * n + x] = P[ly * 1200 + lx]; tid[y * n + x] = i * 3 + j; } }
  const pct = (a, q) => { const s = Float32Array.from(a.filter((_, i) => i % 7 === 0)).sort(); return s[Math.floor(q * (s.length - 1))]; };
  return {tag: c.tag, n, g, r, Pc, tid, S, SG, lo: {gfp: pct(g, 0.002), rfp: pct(r, 0.002)}, hi: {gfp: pct(g, 0.998), rfp: pct(r, 0.998)}, blurCache: {}};
});
function blur(c, sig){
  if (c.blurCache[sig]) return c.blurCache[sig];
  const n = c.n, src = Float32Array.from(c.g);
  if (sig <= 0){ c.blurCache = {[sig]: src}; return src; }
  const rad = Math.ceil(3 * sig), w = []; let ws = 0;
  for (let i = -rad; i <= rad; i++){ const v = Math.exp(-i * i / (2 * sig * sig)); w.push(v); ws += v; }
  for (let i = 0; i < w.length; i++) w[i] /= ws;
  const tmp = new Float32Array(n * n), out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++){ let a = 0;
    for (let i = -rad; i <= rad; i++){ const xx = Math.min(n - 1, Math.max(0, x + i)); a += w[i + rad] * src[y * n + xx]; } tmp[y * n + x] = a; }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++){ let a = 0;
    for (let i = -rad; i <= rad; i++){ const yy = Math.min(n - 1, Math.max(0, y + i)); a += w[i + rad] * tmp[yy * n + x]; } out[y * n + x] = a; }
  c.blurCache = {[sig]: out}; return out;
}
function dropSmall(m, n, val, minSize){
  // set 4-connected components of value `val` smaller than minSize to 1-val
  if (minSize <= 0) return;
  const seen = new Uint8Array(n * n), stack = new Int32Array(n * n), comp = new Int32Array(n * n);
  for (let s = 0; s < n * n; s++){
    if (seen[s] || m[s] !== val) continue;
    let sp = 0, cn = 0; stack[sp++] = s; seen[s] = 1;
    while (sp){ const p = stack[--sp]; comp[cn++] = p; const y = (p / n) | 0, x = p - y * n;
      if (x > 0 && !seen[p-1] && m[p-1] === val){ seen[p-1] = 1; stack[sp++] = p - 1; }
      if (x < n-1 && !seen[p+1] && m[p+1] === val){ seen[p+1] = 1; stack[sp++] = p + 1; }
      if (y > 0 && !seen[p-n] && m[p-n] === val){ seen[p-n] = 1; stack[sp++] = p - n; }
      if (y < n-1 && !seen[p+n] && m[p+n] === val){ seen[p+n] = 1; stack[sp++] = p + n; } }
    if (cn < minSize) for (let i = 0; i < cn; i++) m[comp[i]] = 1 - val;
  }
}
function computeMask(c){
  const n = c.n, sm = blur(c, prm.sm / 10), m = new Uint8Array(n * n), L = prm.level / 100, k = prm.k / 10;
  for (let p = 0; p < n * n; p++){
    const t = c.tid[p];
    const bgv = prm.mode === 'tile' ? L * c.S[t] * c.Pc[p] : L * c.Pc[p];
    const sg = prm.mode === 'tile' ? c.SG[t] : FIXED_SIGMA;
    m[p] = sm[p] > bgv + k * sg ? 1 : 0;
  }
  dropSmall(m, n, 1, prm.speck); dropSmall(m, n, 0, prm.hole);
  return m;
}
function median(arr){ if (!arr.length) return NaN; arr.sort((a, b) => a - b); return arr[arr.length >> 1]; }
function stats(c, m){
  let nf = 0, sg = 0, sr = 0; const bg = [], br = [];
  for (let p = 0; p < m.length; p++){ if (m[p]){ nf++; sg += c.g[p]; sr += c.r[p]; } else if ((p & 7) === 0){ bg.push(c.g[p]); br.push(c.r[p]); } }
  const mg = sg / nf, mr = sr / nf, bgG = median(bg), bgR = median(br);
  return {fg: nf / m.length, mg, mr, ratio: mr / mg, bsub: (mr - bgR) / (mg - bgG), bgG, bgR};
}
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
let curMask = null;
function render(){
  const c = crops[cur], n = c.n, src = chan === 'gfp' ? c.g : c.r, lo = c.lo[chan], hi = c.hi[chan];
  const gm = +document.getElementById('gm').value / 100, op = +document.getElementById('op').value / 100;
  cv.width = n; cv.height = n; const id = cx.createImageData(n, n), d = id.data, m = curMask;
  for (let p = 0; p < n * n; p++){
    let v = (src[p] - lo) / (hi - lo); v = v < 0 ? 0 : v > 1 ? 1 : v; v = 255 * Math.pow(v, gm);
    let R = v, G = v, B = v;
    if (showOv && m){
      if (!m[p]){ R = R * (1 - op) + 40 * op; G = G * (1 - op) + 90 * op; B = B * (1 - op) + 255 * op; }
      else if (showOl){ const y = (p / n) | 0, x = p - y * n;
        if ((x > 0 && !m[p-1]) || (x < n-1 && !m[p+1]) || (y > 0 && !m[p-n]) || (y < n-1 && !m[p+n])){ R = 255; G = 215; B = 0; } }
    }
    d[4*p] = R; d[4*p+1] = G; d[4*p+2] = B; d[4*p+3] = 255;
  }
  cx.putImageData(id, 0, 0);
  const stg = document.getElementById('stage'); stg.style.width = n + 'px'; stg.style.height = n + 'px';
  stg.style.transform = `scale(${zoom})`; stg.style.marginRight = n * (zoom - 1) + 'px'; stg.style.marginBottom = n * (zoom - 1) + 'px';
}
const fmt = (v, d = 0) => isFinite(v) ? v.toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}) : '–';
function rowHTML(c, s, i){ return `<tr data-i="${i}" class="${i === cur ? 'cur' : ''}"><td>${c.tag}</td><td>${fmt(100 * s.fg, 1)}</td><td>${fmt(s.mg)}</td><td>${fmt(s.mr)}</td><td>${fmt(s.ratio, 3)}</td><td>${fmt(s.bsub, 3)}</td><td>${fmt(s.bgG)}</td><td>${fmt(s.bgR)}</td></tr>`; }
let rows = [];
function updateAll(){
  rows = crops.map((c, i) => { const m = computeMask(c); if (i === cur) curMask = m; return rowHTML(c, stats(c, m), i); });
  document.querySelector('#tbl tbody').innerHTML = rows.join('');
  document.querySelectorAll('#tbl tbody tr').forEach(tr => tr.onclick = () => { cur = +tr.dataset.i; curMask = computeMask(crops[cur]); updateAll(); });
  render(); showSettings();
}
function updateCurrent(){
  const c = crops[cur]; curMask = computeMask(c); rows[cur] = rowHTML(c, stats(c, curMask), cur);
  document.querySelector('#tbl tbody').innerHTML = rows.join('');
  document.querySelectorAll('#tbl tbody tr').forEach(tr => tr.onclick = () => { cur = +tr.dataset.i; updateAll(); });
  render(); showSettings();
}
function showSettings(){
  const s = {background_model: prm.mode === 'tile' ? 'per-tile fit' : 'fixed empty-dish', background_level_x: prm.level / 100,
             threshold_noise_units: prm.k / 10, smoothing_px: prm.sm / 10, min_speck_px: prm.speck, max_hole_px: prm.hole};
  document.getElementById('settings').textContent = JSON.stringify(s, null, 1);
  document.getElementById('v_level').textContent = (prm.level / 100).toFixed(2);
  document.getElementById('v_k').textContent = (prm.k / 10).toFixed(1);
  document.getElementById('v_sm').textContent = (prm.sm / 10).toFixed(1);
  document.getElementById('v_speck').textContent = prm.speck; document.getElementById('v_hole').textContent = prm.hole;
  document.getElementById('v_gm').textContent = (+document.getElementById('gm').value / 100).toFixed(2);
  document.getElementById('v_op').textContent = document.getElementById('op').value + '%';
}
let pending = false;
['level', 'k', 'sm', 'speck', 'hole'].forEach(id => {
  const el = document.getElementById(id);
  el.oninput = () => { prm[id] = +el.value; showSettings(); if (!pending){ pending = true; requestAnimationFrame(() => { pending = false; updateCurrent(); }); } };
  el.onchange = () => updateAll();
});
['gm', 'op'].forEach(id => document.getElementById(id).oninput = () => { showSettings(); render(); });
function setMode(mo){ prm.mode = mo; document.getElementById('m_tile').classList.toggle('on', mo === 'tile'); document.getElementById('m_fixed').classList.toggle('on', mo === 'fixed'); updateAll(); }
document.getElementById('m_tile').onclick = () => setMode('tile');
document.getElementById('m_fixed').onclick = () => setMode('fixed');
function setChan(ch){ chan = ch; document.getElementById('b_gfp').classList.toggle('on', ch === 'gfp'); document.getElementById('b_rfp').classList.toggle('on', ch === 'rfp'); render(); }
document.getElementById('b_gfp').onclick = () => setChan('gfp'); document.getElementById('b_rfp').onclick = () => setChan('rfp');
function toggleOv(){ showOv = !showOv; document.getElementById('ov').classList.toggle('on', showOv); render(); }
document.getElementById('ov').onclick = toggleOv;
document.getElementById('ol').onclick = () => { showOl = !showOl; document.getElementById('ol').classList.toggle('on', showOl); render(); };
document.querySelectorAll('.zb').forEach(b => b.onclick = () => { zoom = +b.dataset.z; document.querySelectorAll('.zb').forEach(x => x.classList.toggle('on', x === b)); render(); });
document.getElementById('reset').onclick = () => { prm = Object.assign({}, DEF); ['level', 'k', 'sm', 'speck', 'hole'].forEach(id => document.getElementById(id).value = prm[id]); setMode('tile'); };
document.getElementById('copy').onclick = () => { try { navigator.clipboard.writeText(document.getElementById('settings').textContent); } catch(e) {} };
window.addEventListener('keydown', e => { if (e.key.toLowerCase() === 'h' && e.target.tagName !== 'INPUT') toggleOv(); });
updateAll();
</script></body></html>
"""


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a.astype("<u2")).tobytes()).decode()


def main():
    data_dir, out = sys.argv[1], sys.argv[2]
    crops = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f)
        crops.append(dict(tag=os.path.basename(f)[:-4], n=int(z["gfp"].shape[0]), y0=int(z["y0"]), x0=int(z["x0"]),
                          gfp=b64(z["gfp"]), rfp=b64(z["rfp"]),
                          tiles=[[int(t[0]), int(t[1]), int(t[2]), float(t[3]), float(t[4])] for t in z["tiles"]]))
    data = dict(P=b64(np.load(os.path.join(data_dir, "P.npy"))), crops=crops)
    with open(out, "w") as fh:
        fh.write(TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {out} with {len(crops)} crops")


if __name__ == "__main__":
    main()
