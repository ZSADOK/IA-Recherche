# svg.py
from __future__ import annotations
from typing import Tuple
from core.genotype import Genotype


def export_svg(path: str, genotype: Genotype, width: int, height: int, background_bgr: Tuple[int, int, int]) -> None:
    b, g, r = background_bgr
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        )
        f.write(f'  <rect width="100%" height="100%" fill="rgb({r},{g},{b})"/>\n')
        for s in genotype.shapes:
            f.write("  " + s.to_svg() + "\n")
        f.write("</svg>\n")
