"""Pages of the panel cut sheet: flat sheets packed in shelves onto stock-sized pages. Pure
numbers in millimetres; a sheet bigger than the stock gets a page of its own, sized to it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    """Where one sheet goes: page index (from 0), its top-left corner on that page [mm]."""

    page: int
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class Page:
    """A page of the cut sheet: size [mm] and whether it is a stock page or an oversize one."""

    width_mm: float
    height_mm: float
    oversize: bool


def pack_shelves(
    sizes_mm: list[tuple[float, float]], stock_mm: tuple[float, float], gap_mm: float
) -> tuple[list[Placement], list[Page]]:
    """Shelf packing, tallest first: sheets go left to right in rows on the first stock page with
    room (a row is as tall as its tallest sheet). A sheet that does not fit the stock page on
    its own gets an oversize page of its size plus the gap. Placements are in input order."""
    stock_w, stock_h = stock_mm
    order = sorted(range(len(sizes_mm)), key=lambda i: (-sizes_mm[i][1], -sizes_mm[i][0], i))
    pages: list[Page] = []
    shelves: dict[int, list[list[float]]] = {}  # stock page -> [[y, row height, x cursor], ...]
    placed: dict[int, Placement] = {}
    for i in order:
        w, h = sizes_mm[i]
        if w + 2 * gap_mm > stock_w or h + 2 * gap_mm > stock_h:
            pages.append(Page(w + 2 * gap_mm, h + 2 * gap_mm, True))
            placed[i] = Placement(len(pages) - 1, gap_mm, gap_mm)
            continue
        spot = _first_fit(pages, shelves, w, h, stock_mm, gap_mm)
        placed[i] = spot
    return [placed[i] for i in range(len(sizes_mm))], pages


def _first_fit(
    pages: list[Page],
    shelves: dict[int, list[list[float]]],
    w: float,
    h: float,
    stock_mm: tuple[float, float],
    gap_mm: float,
) -> Placement:
    """Place a w x h sheet on the first stock page whose shelves or free height take it."""
    stock_w, stock_h = stock_mm
    for page, rows in shelves.items():
        for row in rows:
            if h <= row[1] and row[2] + w + gap_mm <= stock_w:
                x, row[2] = row[2], row[2] + w + gap_mm
                return Placement(page, x, row[0])
        used = rows[-1][0] + rows[-1][1] + gap_mm
        if used + h + gap_mm <= stock_h and w + 2 * gap_mm <= stock_w:
            rows.append([used, h, gap_mm + w + gap_mm])
            return Placement(page, gap_mm, used)
    pages.append(Page(stock_w, stock_h, False))
    shelves[len(pages) - 1] = [[gap_mm, h, gap_mm + w + gap_mm]]
    return Placement(len(pages) - 1, gap_mm, gap_mm)
