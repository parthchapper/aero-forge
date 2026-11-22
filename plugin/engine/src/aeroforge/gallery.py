"""Static gallery HTML: tiles (iso + top), click to mark good/bad, export marks.json."""

import html
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """One gallery entry; image paths are relative to gallery.html; error set on a failed build.

    score is the style score (top-by-style galleries only); it is exported, never shown.
    caption and rows (check id, passed, margin) describe a search winner; every row is shown.
    """

    tile_id: str
    genome: tuple[float, ...]
    iso_path: str | None
    top_path: str | None
    error: str | None
    score: float | None = None
    caption: str = ""
    rows: tuple[tuple[str, bool, float], ...] = ()


STYLE = """
body{margin:0;padding:16px;font:14px/1.4 system-ui,sans-serif;background:#f4f5f7;color:#1d2330}
header{position:sticky;top:0;background:#f4f5f7;padding:8px 0 12px;z-index:1;
  display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid #9aa3b2;
  background:#fff;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,652px),1fr));gap:12px}
.tile{background:#fff;border:4px solid #fff;border-radius:8px;padding:4px;
  cursor:pointer;user-select:none}
.tile.good{border-color:#2e9b57}.tile.bad{border-color:#c8433a}
.imgs{display:flex;flex-wrap:wrap}.imgs img{width:50%;max-width:320px;height:auto;display:block}
.label{display:flex;justify-content:space-between;padding:2px 6px;color:#566074}
.err{padding:40px 12px;color:#c8433a;min-height:200px}
.cap{padding:2px 6px;color:#1d2330}
.checks{border-collapse:collapse;margin:4px 6px;font-size:12px}
.checks td{padding:1px 8px 1px 0}.checks .fail{color:#c8433a;font-weight:600}
"""

SCRIPT = """
const data = JSON.parse(document.getElementById('data').textContent);
const marks = {};
const order = ['none', 'good', 'bad'];
function count() {
  const v = Object.values(marks);
  document.getElementById('count').textContent =
    `${v.filter(m => m === 'good').length} good · ` +
    `${v.filter(m => m === 'bad').length} bad · ${data.items.length} tiles`;
}
document.querySelectorAll('.tile').forEach(tile => {
  tile.addEventListener('click', () => {
    const id = tile.dataset.id;
    const next = order[(order.indexOf(marks[id] || 'none') + 1) % order.length];
    marks[id] = next;
    tile.classList.remove('good', 'bad');
    if (next !== 'none') tile.classList.add(next);
    tile.querySelector('.mark').textContent = next;
    count();
  });
});
document.getElementById('export').addEventListener('click', () => {
  const items = data.items.map(
    it => ({...it, mark: marks[it.id] || 'none'}));
  const text = JSON.stringify({seed: data.seed, brief: data.brief, fin_cap: true, items}, null, 1);
  const blob = new Blob([text], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `marks_${data.brief}_seed${data.seed}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});
count();
"""


def _tile_html(tile: Tile) -> str:
    """Markup of one tile."""
    tid = html.escape(tile.tile_id)
    if tile.error is not None:
        body = f'<div class="err">build failed: {html.escape(tile.error)}</div>'
    else:
        size = 'width="320" height="320"'
        iso, top = html.escape(str(tile.iso_path)), html.escape(str(tile.top_path))
        body = (
            f'<div class="imgs"><img src="{iso}" alt="{tid} iso" {size}>'
            f'<img src="{top}" alt="{tid} top" {size}></div>'
        )
    label = f'<div class="label"><span>{tid}</span><span class="mark">none</span></div>'
    return f'<div class="tile" data-id="{tid}">{body}{label}{_checks_html(tile)}</div>'


def _checks_html(tile: Tile) -> str:
    """Caption and the check table of a winner tile (FAIL rows marked, none hidden)."""
    if not tile.caption and not tile.rows:
        return ""
    rows = "".join(
        f'<tr class="{"pass" if ok else "fail"}"><td>{html.escape(cid)}</td>'
        f"<td>{'PASS' if ok else 'FAIL'}</td><td>{margin:+.3f}</td></tr>"
        for cid, ok, margin in tile.rows
    )
    return f'<div class="cap">{html.escape(tile.caption)}</div><table class="checks">{rows}</table>'


def _item(tile: Tile) -> dict[str, object]:
    """Export payload of one tile (marks.json item without the mark)."""
    item: dict[str, object] = {"id": tile.tile_id, "genome": list(tile.genome)}
    if tile.score is not None:
        item["score"] = tile.score
    return item


def gallery_html(tiles: list[Tile], seed: int, brief: str) -> str:
    """Self-contained gallery page (images by relative path, works from file://)."""
    payload = {"seed": seed, "brief": brief, "items": [_item(t) for t in tiles]}
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    cards = "\n".join(_tile_html(t) for t in tiles)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>aero-forge gallery</title><style>{STYLE}</style></head><body>"
        f"<header><h1>aero-forge gallery · {html.escape(brief)} · seed {seed}</h1>"
        '<span id="count"></span>'
        "<span>click a tile: good → bad → none</span>"
        '<button id="export">Export marks.json</button></header>'
        f'<div class="grid">\n{cards}\n</div>'
        f'<script id="data" type="application/json">{data}</script>'
        f"<script>{SCRIPT}</script></body></html>\n"
    )
