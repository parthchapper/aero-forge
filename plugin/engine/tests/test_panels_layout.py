"""U2 cut sheet pagination: every sheet inside a page, no overlaps, oversize sheets flagged."""

import re

import numpy as np
import pytest
from conftest import u2_random_genomes

from aeroforge.brief import brief_requirements
from aeroforge.genetable import pin_choice
from aeroforge.genome import decode
from aeroforge.panels import body_panels, design_panels, panels_svg
from aeroforge.panels_layout import pack_shelves

STOCK = (1200.0, 800.0)
GAP = 12.0


def test_shelf_packing_keeps_sheets_inside_the_page_and_apart() -> None:
    rng = np.random.default_rng(3)
    sizes = [(float(w), float(h)) for w, h in rng.uniform(20, 600, size=(80, 2))]
    spots, pages = pack_shelves(sizes, STOCK, GAP)
    assert len(spots) == len(sizes) and all(not p.oversize for p in pages)
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for (w, h), s in zip(sizes, spots, strict=True):
        page = pages[s.page]
        assert s.x_mm >= 0 and s.y_mm >= 0
        assert s.x_mm + w <= page.width_mm + 1e-9 and s.y_mm + h <= page.height_mm + 1e-9
        for ox, oy, ow, oh in boxes.setdefault(s.page, []):
            assert s.x_mm >= ox + ow or ox >= s.x_mm + w or s.y_mm >= oy + oh or oy >= s.y_mm + h
        boxes[s.page].append((s.x_mm, s.y_mm, w, h))


def test_a_sheet_larger_than_the_stock_gets_an_oversize_page_of_its_own() -> None:
    spots, pages = pack_shelves([(100.0, 100.0), (2000.0, 300.0)], STOCK, GAP)
    assert [p.oversize for p in pages] == [True, False] or [p.oversize for p in pages] == [
        False,
        True,
    ]
    big = pages[spots[1].page]
    assert big.oversize and big.width_mm >= 2000.0 + 2 * GAP
    assert spots[0].page != spots[1].page


def _svg_polygons(svg: str) -> list[np.ndarray]:
    return [
        np.array([[float(t) for t in q.split(",")] for q in m.split()])
        for m in re.findall(r'<polygon points="([^"]+)"', svg)
    ]


@pytest.mark.parametrize("seed", [1234, 99])
def test_the_svg_viewbox_holds_every_sheet_of_large_and_small_bodies(seed: int) -> None:
    req = brief_requirements("stub")
    gs = pin_choice(u2_random_genomes(seed, 12), "body_style", "faceted")
    for g in gs:
        d = decode(g, req)
        panels = design_panels(d)
        if not panels:
            continue
        svg = panels_svg(panels)
        vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        assert vb is not None
        width, height = float(vb.group(1)), float(vb.group(2))
        polys = _svg_polygons(svg)
        assert len(polys) == len(panels)
        for poly in polys:
            assert poly[:, 0].max() <= width + 1e-6 and poly[:, 1].max() <= height + 1e-6
        pages = re.findall(r'<g id="page-(\d+)">', svg)
        assert len(pages) >= 1 and "Sheet 1 of " in svg
        for p in body_panels(d.bodies[0]) if d.bodies[0].facets else []:
            assert f'id="{p.id}"' in svg
