"""viewer.html (SPEC §11): self-contained three.js viewer of assembly.glb with the part table.

three.js, GLTFLoader and OrbitControls come from cdn.jsdelivr.net through an importmap; the GLB
is embedded as base64, so the page opens from file://. Clicking a part (or its table row) shows
its name and the Design fields behind it."""

import base64
import html
import json

from aeroforge.constants import constant
from aeroforge.internals import Internal
from aeroforge.mesh import part_kind
from aeroforge.models import Design

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body{margin:0;font:13px/1.4 system-ui,sans-serif;background:#f4f5f7;color:#16181d;
  display:flex;height:100vh}
#view{flex:1;min-width:0;position:relative}
#side{width:360px;overflow:auto;border-left:1px solid #d5d8de;background:#fff}
h1{font-size:15px;margin:12px}h2{font-size:13px;margin:12px 12px 4px}
table{border-collapse:collapse;width:100%}
td,th{padding:3px 12px;text-align:left;border-bottom:1px solid #eceef1}
tr.part{cursor:pointer}tr.part:hover,tr.sel{background:#e4ecfb}
pre{margin:0 12px 12px;white-space:pre-wrap;word-break:break-word;font-size:12px}
#hint{position:absolute;left:12px;bottom:10px;color:#5b6170}
</style></head><body>
<div id="view"><div id="hint">drag: orbit · wheel: zoom · click a part (internals first)
or a table row</div></div>
<div id="side"><h1>__TITLE__</h1><h2 id="selname">no part selected</h2><pre id="fields"></pre>
<h2>Parts</h2><table id="table"><tr><th>part</th><th>kind</th></tr>__ROWS__</table></div>
<script id="glb" type="application/octet-stream">__GLB__</script>
<script id="parts" type="application/json">__PARTS__</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@__V__/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@__V__/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const parts = JSON.parse(document.getElementById("parts").textContent);
const colours = __COLOURS__;
const raw = atob(document.getElementById("glb").textContent.trim());
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const view = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f5f7);
scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-1, -1, 2); scene.add(sun);
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
const meshes = {};
let selected = null;
function select(name) {
  if (selected && meshes[selected]) meshes[selected].material.emissive.setHex(0x000000);
  selected = name in parts ? name : null;
  document.querySelectorAll("tr.part").forEach(
    r => r.classList.toggle("sel", r.dataset.name === selected));
  document.getElementById("selname").textContent = selected || "no part selected";
  document.getElementById("fields").textContent =
    selected ? JSON.stringify(parts[selected], null, 1) : "";
  if (selected && meshes[selected]) meshes[selected].material.emissive.setHex(0x3355aa);
}
new GLTFLoader().parse(bytes.buffer, "", gltf => {
  gltf.scene.traverse(o => {
    if (!o.isMesh) return;
    let n = o; while (n && !(n.name in parts)) n = n.parent;
    if (!n) return;
    o.geometry.computeVertexNormals();  // the GLB carries positions only
    const c = colours[parts[n.name].kind] || [0.7, 0.7, 0.7];
    o.material = new THREE.MeshStandardMaterial({color: new THREE.Color(...c), roughness: 0.6,
      transparent: parts[n.name].kind !== "internal",
      opacity: parts[n.name].kind === "internal" ? 1 : 0.55,
      side: THREE.DoubleSide});
    o.userData.part = n.name; meshes[n.name] = o;
  });
  scene.add(gltf.scene);
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const size = box.getSize(new THREE.Vector3()).length(), mid = box.getCenter(new THREE.Vector3());
  controls.target.copy(mid);
  camera.position.copy(mid).add(new THREE.Vector3(-0.8, -1.0, 0.7).multiplyScalar(size));
  controls.update();
}, err => { document.getElementById("selname").textContent = "GLB load failed: " + err; });
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
let down = null;
renderer.domElement.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", e => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(ptr, camera);
  // the skin is translucent: a click selects the first internal behind it, else the skin
  const hits = ray.intersectObjects(Object.values(meshes), false);
  const inner = hits.find(h => parts[h.object.userData.part].kind === "internal");
  const hit = inner || hits[0];
  select(hit ? hit.object.userData.part : null);
});
document.querySelectorAll("tr.part").forEach(
  r => r.addEventListener("click", () => select(r.dataset.name)));
function resize() {
  renderer.setSize(view.clientWidth, view.clientHeight);
  camera.aspect = view.clientWidth / view.clientHeight; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def _base(name: str) -> str:
    """Component name of a part instance (`surface0_L` -> `surface0`)."""
    return name[:-2] if name.endswith(("_L", "_R")) else name


def part_table(
    design: Design, items: tuple[Internal, ...], names: list[str]
) -> dict[str, dict[str, object]]:
    """Per part instance: kind, the instance side and the Design (or internal) fields behind it."""
    fields: dict[str, dict[str, object]] = {}
    for group in (design.surfaces, design.bodies, design.propulsors):
        for c in group:
            fields[c.name] = c.model_dump(mode="json")
    for it in items:
        fields[it.name] = it.model_dump(mode="json")
    table = {}
    for n in sorted(names):
        side = {"_L": "left", "_R": "right"}.get(n[-2:], "centre")
        table[n] = {"kind": part_kind(n), "instance": side, "fields": fields.get(_base(n), {})}
    return table


def _script_json(payload: object) -> str:
    """JSON safe inside a <script> element."""
    return json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")


def viewer_html(title: str, glb: bytes, table: dict[str, dict[str, object]]) -> str:
    """Self-contained viewer page: base64 GLB, part table, three.js from cdn.jsdelivr.net."""
    rows = "".join(
        f'<tr class="part" data-name="{html.escape(n)}"><td>{html.escape(n)}</td>'
        f"<td>{html.escape(str(row['kind']))}</td></tr>"
        for n, row in table.items()
    )
    return (
        PAGE.replace("__TITLE__", html.escape(title))
        .replace("__ROWS__", rows)
        .replace("__V__", str(constant("constants", "viewer_three_version")))
        .replace("__COLOURS__", _script_json(constant("constants", "render_colours")))
        .replace("__PARTS__", _script_json(table))
        .replace("__GLB__", base64.b64encode(glb).decode("ascii"))
    )
