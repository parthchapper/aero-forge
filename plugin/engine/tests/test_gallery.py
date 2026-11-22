import json
import re
from pathlib import Path

import pytest

from aeroforge.cli import main
from aeroforge.gallery import Tile, gallery_html


def _tiles(n: int, score: float | None = None) -> list[Tile]:
    return [
        Tile(
            f"g{i:03d}",
            (0.5, 0.25),
            f"tiles/g{i:03d}_iso.png",
            f"tiles/g{i:03d}_top.png",
            None,
            score,
        )
        for i in range(n)
    ]


def _payload(page: str) -> dict[str, object]:
    return json.loads(
        re.search(r'<script id="data" type="application/json">(.*?)</script>', page).group(1)
    )


def test_gallery_has_one_tile_per_entry() -> None:
    page = gallery_html(_tiles(3), seed=1, brief="stub")
    assert len(re.findall(r'class="tile"', page)) == 3


def test_gallery_embeds_genomes_for_marks_export() -> None:
    page = gallery_html(_tiles(2), seed=7, brief="small_boxes")
    assert _payload(page) == {
        "seed": 7,
        "brief": "small_boxes",
        "items": [{"id": "g000", "genome": [0.5, 0.25]}, {"id": "g001", "genome": [0.5, 0.25]}],
    }


def test_scored_tiles_export_their_score_but_do_not_show_it() -> None:
    page = gallery_html(_tiles(1, score=0.8125), seed=7, brief="stub")
    assert _payload(page)["items"][0]["score"] == 0.8125
    assert "0.8125" not in page.split('<script id="data"')[0]


def test_failed_tile_shows_reason_and_escapes_markup() -> None:
    tile = Tile("g000", (0.1,), None, None, "GeometryError: <bad>")
    page = gallery_html([tile], seed=1, brief="stub")
    assert "build failed: GeometryError: &lt;bad&gt;" in page


def test_cli_rejects_non_positive_n(tmp_path: Path) -> None:
    assert main(["gallery", "--n", "0", "--seed", "1", "--out", str(tmp_path)]) == 2


def test_cli_rejects_an_unknown_brief(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["gallery", "--n", "1", "--seed", "1", "--brief", "nope", "--out", str(tmp_path)])
    assert exc.value.code == 2


@pytest.mark.slow
@pytest.mark.parametrize("brief", ["stub", "small_boxes"])
def test_cli_gallery_writes_page_and_tiles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], brief: str
) -> None:
    args = ["gallery", "--n", "2", "--seed", "1", "--brief", brief, "--out", str(tmp_path)]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["brief"] == brief
    assert summary["built"] + len(summary["failed"]) == 2
    assert (tmp_path / "gallery.html").is_file()
    assert len(list((tmp_path / "tiles").glob("*.png"))) == 2 * summary["built"]


@pytest.mark.slow
def test_cli_top_by_style_gallery_exports_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["gallery", "--n", "2", "--seed", "3", "--top-by-style", "--out", str(tmp_path)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["top_by_style"] is True
    items = _payload((tmp_path / "gallery.html").read_text(encoding="utf-8"))["items"]
    assert all(0.0 < it["score"] < 1.0 for it in items)
