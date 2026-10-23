#!/usr/bin/env python
"""Build a self-contained HTML page for painting background / cell scribbles on GFP crops.

Usage:
    python make_label_page.py <crop_dir> <out.html>

<crop_dir> holds *_GFP_raw.tif (uint16, any size). The raw counts are embedded as base64 so
the page can re-stretch contrast live (dim cells vs. dark dish is the hard call). Labels are
exported as PNGs (red channel: 0 = unlabeled, 1 = background, 2 = cell) named
<tag>_labels.png, which land in the browser's download folder.
"""
import base64
import glob
import json
import os
import sys

import numpy as np
import tifffile

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Background Labeling</title>
<style>
:root{--bg:#16181d;--panel:#20232a;--text:#e6e6e6;--muted:#9aa0aa;--accent:#4c8dff;--bgc:#2f6eff;--cellc:#ffbe0b}
body{margin:0;background:var(--bg);color:var(--text);font:14px -apple-system,system-ui,sans-serif}
header{padding:10px 16px;background:var(--panel);display:flex;flex-wrap:wrap;gap:14px;align-items:center;position:sticky;top:0;z-index:5}
button{background:#2c313a;color:var(--text);border:1px solid #3a404b;border-radius:6px;padding:6px 10px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.grp{display:flex;gap:6px;align-items:center}
label{color:var(--muted)}
#wrap{overflow:auto;height:calc(100vh - 120px);margin:0 16px}
#stage{position:relative;transform-origin:0 0}
#stage canvas{position:absolute;left:0;top:0;image-rendering:pixelated}
#help{padding:6px 16px;color:var(--muted);font-size:13px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:4px}
</style></head><body>
<header>
  <div class="grp" id="tabs"></div>
  <div class="grp"><button id="t_bg" class="on"><span class="sw" style="background:var(--bgc)"></span>Background (B)</button>
  <button id="t_cell"><span class="sw" style="background:var(--cellc)"></span>Cell (C)</button>
  <button id="t_er">Erase (E)</button></div>
  <div class="grp"><label>Brush</label><input id="brush" type="range" min="2" max="80" value="18"><span id="bv">18</span></div>
  <div class="grp"><label>Zoom</label><button data-z="1" class="zb on">1×</button><button data-z="2" class="zb">2×</button><button data-z="3" class="zb">3×</button></div>
  <div class="grp"><label>Black</label><input id="lo" type="range" min="0" max="1000" value="0">
  <label>White</label><input id="hi" type="range" min="0" max="1000" value="1000">
  <label>Gamma</label><input id="gm" type="range" min="20" max="150" value="60"></div>
  <div class="grp"><button id="hide">Hide labels (H)</button><button id="undo">Undo (Z)</button>
  <button id="save" style="background:#1f7a3a;border-color:#1f7a3a">Save all labels</button></div>
</header>
<div id="help">Paint <b>background</b> (empty dish) wherever you are confident; include the hard dark gaps between cells.
Optionally paint <b>cells</b>, especially dim ones near background. Scribbles are fine; you don't need to cover everything.
Labels autosave in this browser. "Save all labels" downloads one PNG per image. <span id="stat"></span></div>
<div id="wrap"><div id="stage"><canvas id="img"></canvas><canvas id="lab"></canvas></div></div>
<script>
const IMAGES = __IMAGES__;
const COL = {1:[47,110,255], 2:[255,190,11]};
let cur = 0, tool = 1, zoom = 1, hidden = false, painting = false;
const st = IMAGES.map(im => {
  const raw = atob(im.data), u8 = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);
  const px = new Uint16Array(u8.buffer);
  const sorted = Float64Array.from(px).sort();
  const pct = q => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return {px, w: im.w, h: im.h, tag: im.tag, p0: pct(0.001), p100: pct(0.999),
          lab: load(im.tag, im.w * im.h), undo: [], lo: 0, hi: 1000, gm: 60};
});
function load(tag, n){
  try { const s = localStorage.getItem('fgbg_' + tag); if (s) return unrle(JSON.parse(s), n); } catch(e) {}
  return new Uint8Array(n);
}
function persist(s){ try { localStorage.setItem('fgbg_' + s.tag, JSON.stringify(rle(s.lab))); } catch(e) {} }
function rle(a){ const o = []; let v = a[0], n = 0; for (const x of a){ if (x === v) n++; else { o.push(v, n); v = x; n = 1; } } o.push(v, n); return o; }
function unrle(r, n){ const a = new Uint8Array(n); let i = 0; for (let k = 0; k < r.length; k += 2){ a.fill(r[k], i, i + r[k+1]); i += r[k+1]; } return a; }
const ci = document.getElementById('img'), cl = document.getElementById('lab');
const xi = ci.getContext('2d'), xl = cl.getContext('2d');
function renderImg(){
  const s = st[cur], span = s.p100 - s.p0;
  const lo = s.p0 + span * s.lo / 1000, hi = s.p0 + span * s.hi / 1000, g = s.gm / 100;
  const id = xi.createImageData(s.w, s.h), d = id.data;
  for (let i = 0; i < s.px.length; i++){
    let v = (s.px[i] - lo) / Math.max(1, hi - lo); v = v < 0 ? 0 : v > 1 ? 1 : v;
    const c = 255 * Math.pow(v, g); d[4*i] = d[4*i+1] = d[4*i+2] = c; d[4*i+3] = 255;
  }
  xi.putImageData(id, 0, 0);
}
function renderLab(x0 = 0, y0 = 0, x1 = null, y1 = null){
  const s = st[cur]; x1 = x1 ?? s.w; y1 = y1 ?? s.h;
  x0 = Math.max(0, x0|0); y0 = Math.max(0, y0|0); x1 = Math.min(s.w, Math.ceil(x1)); y1 = Math.min(s.h, Math.ceil(y1));
  if (x1 <= x0 || y1 <= y0) return;
  const id = xl.createImageData(x1 - x0, y1 - y0), d = id.data;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++){
    const v = s.lab[y * s.w + x], o = 4 * ((y - y0) * (x1 - x0) + (x - x0));
    if (v){ const c = COL[v]; d[o] = c[0]; d[o+1] = c[1]; d[o+2] = c[2]; d[o+3] = 115; }
  }
  xl.putImageData(id, x0, y0);
}
function stat(){
  const s = st[cur]; let b = 0, c = 0; for (const v of s.lab){ if (v === 1) b++; else if (v === 2) c++; }
  document.getElementById('stat').textContent = `· ${s.tag}: ${b.toLocaleString()} bg px, ${c.toLocaleString()} cell px labeled`;
}
function show(i){
  cur = i; const s = st[i];
  [ci, cl].forEach(c => { c.width = s.w; c.height = s.h; });
  const stg = document.getElementById('stage'); stg.style.width = s.w + 'px'; stg.style.height = s.h + 'px';
  setZoom(zoom);
  ['lo', 'hi', 'gm'].forEach(k => document.getElementById(k).value = s[k]);
  document.querySelectorAll('#tabs button').forEach((b, k) => b.classList.toggle('on', k === i));
  renderImg(); xl.clearRect(0, 0, s.w, s.h); renderLab(); stat();
}
function setZoom(z){
  zoom = z; const s = st[cur], stg = document.getElementById('stage');
  stg.style.transform = `scale(${z})`; stg.parentElement.style.setProperty('--z', z);
  stg.style.marginRight = (s.w * (z - 1)) + 'px'; stg.style.marginBottom = (s.h * (z - 1)) + 'px';
  document.querySelectorAll('.zb').forEach(b => b.classList.toggle('on', +b.dataset.z === z));
}
let last = null;
function dab(ev){
  // fill the gap since the previous mouse event so fast strokes stay continuous
  const r = cl.getBoundingClientRect();
  const x = (ev.clientX - r.left) / zoom, y = (ev.clientY - r.top) / zoom;
  const rad = +document.getElementById('brush').value / 2;
  if (last){
    const n = Math.ceil(Math.hypot(x - last[0], y - last[1]) / Math.max(1, rad / 2));
    for (let k = 1; k < n; k++) stamp(last[0] + (x - last[0]) * k / n, last[1] + (y - last[1]) * k / n, rad);
  }
  stamp(x, y, rad); last = [x, y];
}
function stamp(x, y, rad){
  const s = st[cur], val = tool === 0 ? 0 : tool;
  for (let yy = Math.max(0, Math.floor(y - rad)); yy <= Math.min(s.h - 1, y + rad); yy++)
    for (let xx = Math.max(0, Math.floor(x - rad)); xx <= Math.min(s.w - 1, x + rad); xx++)
      if ((xx - x) ** 2 + (yy - y) ** 2 <= rad * rad) s.lab[yy * s.w + xx] = val;
  xl.clearRect(x - rad - 1, y - rad - 1, 2 * rad + 3, 2 * rad + 3);
  renderLab(x - rad - 1, y - rad - 1, x + rad + 2, y + rad + 2);
}
cl.addEventListener('mousedown', e => { const s = st[cur]; s.undo.push(s.lab.slice()); if (s.undo.length > 25) s.undo.shift(); painting = true; last = null; dab(e); });
window.addEventListener('mousemove', e => { if (painting) dab(e); });
window.addEventListener('mouseup', () => { if (painting){ painting = false; last = null; persist(st[cur]); stat(); } });
function setTool(t){ tool = t; [['t_er', 0], ['t_bg', 1], ['t_cell', 2]].forEach(([id, v]) => document.getElementById(id).classList.toggle('on', v === t)); }
document.getElementById('t_bg').onclick = () => setTool(1);
document.getElementById('t_cell').onclick = () => setTool(2);
document.getElementById('t_er').onclick = () => setTool(0);
document.getElementById('brush').oninput = e => document.getElementById('bv').textContent = e.target.value;
document.querySelectorAll('.zb').forEach(b => b.onclick = () => setZoom(+b.dataset.z));
['lo', 'hi', 'gm'].forEach(k => document.getElementById(k).oninput = e => { st[cur][k] = +e.target.value; renderImg(); });
function toggleHide(){ hidden = !hidden; cl.style.display = hidden ? 'none' : ''; document.getElementById('hide').classList.toggle('on', hidden); }
document.getElementById('hide').onclick = toggleHide;
function undo(){ const s = st[cur]; if (s.undo.length){ s.lab = s.undo.pop(); xl.clearRect(0, 0, s.w, s.h); renderLab(); persist(s); stat(); } }
document.getElementById('undo').onclick = undo;
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 'b') setTool(1); else if (k === 'c') setTool(2); else if (k === 'e') setTool(0);
  else if (k === 'h') toggleHide(); else if (k === 'z') undo();
  else if (k === '[' || k === ']'){ const b = document.getElementById('brush'); b.value = +b.value + (k === ']' ? 4 : -4); b.oninput({target: b}); }
});
document.getElementById('save').onclick = () => {
  st.forEach((s, k) => setTimeout(() => {
    const c = document.createElement('canvas'); c.width = s.w; c.height = s.h;
    const x = c.getContext('2d'), id = x.createImageData(s.w, s.h);
    for (let i = 0; i < s.lab.length; i++){ id.data[4*i] = s.lab[i]; id.data[4*i+3] = 255; }
    x.putImageData(id, 0, 0);
    const a = document.createElement('a'); a.download = s.tag + '_labels.png'; a.href = c.toDataURL('image/png'); a.click();
  }, k * 400));
};
IMAGES.forEach((im, i) => { const b = document.createElement('button'); b.textContent = im.tag; b.onclick = () => show(i); document.getElementById('tabs').appendChild(b); });
show(0);
</script></body></html>
"""


def main():
    crop_dir, out = sys.argv[1], sys.argv[2]
    images = []
    for f in sorted(glob.glob(os.path.join(crop_dir, "*_GFP_raw.tif"))):
        a = tifffile.imread(f).astype("<u2")
        tag = os.path.basename(f).replace("_GFP_raw.tif", "")
        images.append(dict(tag=tag, w=int(a.shape[1]), h=int(a.shape[0]),
                           data=base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()))
    with open(out, "w") as fh:
        fh.write(TEMPLATE.replace("__IMAGES__", json.dumps(images)))
    print(f"wrote {out} with {len(images)} images")


if __name__ == "__main__":
    main()
