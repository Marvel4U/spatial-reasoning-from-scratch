"""Patch / grid geometry for c0 viewer (image px ↔ output grid)."""
from __future__ import annotations

from dataclasses import dataclass

from plain_gpt_module.local_config import LocalGridViTConfig


@dataclass(frozen=True)
class C0Geometry:
    patch_size: int
    img_size: int
    grid_size: int
    n_patch: int
    subcells: int

    @classmethod
    def from_config(cls, cfg: LocalGridViTConfig) -> C0Geometry:
        return cls(
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            grid_size=cfg.grid_out_size,
            n_patch=cfg.patch_grid,
            subcells=cfg.subcells,
        )

    def marker_grid_cell(self, marker_row: int, marker_col: int) -> tuple[int, int]:
        g, h = self.grid_size, self.img_size
        return marker_row * g // h, marker_col * g // h

    def patch_index(self, marker_row: int, marker_col: int) -> tuple[int, int]:
        p = self.patch_size
        return marker_row // p, marker_col // p

    def patch_pixel_bounds(self, pr: int, pc: int) -> tuple[slice, slice]:
        p = self.patch_size
        return slice(pr * p, (pr + 1) * p), slice(pc * p, (pc + 1) * p)

    def patch_block(self, gr: int, gc: int) -> tuple[int, int]:
        s = self.subcells
        return gr // s, gc // s

    def subcell_in_patch(self, gr: int, gc: int) -> tuple[int, int]:
        s = self.subcells
        return gr % s, gc % s

    def patch_block_bounds(self, br: int, bc: int) -> tuple[slice, slice]:
        s = self.subcells
        return slice(br * s, (br + 1) * s), slice(bc * s, (bc + 1) * s)

    def neighborhood_blocks(
        self, br: int, bc: int, radius: int = 1,
    ) -> tuple[slice, slice]:
        n = self.n_patch
        r0 = max(0, br - radius)
        r1 = min(n, br + radius + 1)
        c0 = max(0, bc - radius)
        c1 = min(n, bc + radius + 1)
        s = self.subcells
        return slice(r0 * s, r1 * s), slice(c0 * s, c1 * s)
