# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Custom terrain generation functions for stair climbing training and evaluation.

Training terrains (both ends at z=0, seamless tiling):
    - TrapezoidStairsTerrainCfg:         flat(z=0) → UP → plateau → DOWN → flat(z=0)
    - InvertedTrapezoidStairsTerrainCfg:  flat(z=0) → DOWN → valley(z<0) → UP → flat(z=0)

Evaluation terrains (one-way, for measuring directional performance):
    - StraightAscendingStairsTerrainCfg:  flat → UP → flat (elevated)
    - StraightDescendingStairsTerrainCfg: flat (elevated) → DOWN → flat

All training terrains have edges at z=0 and use the full cell size (W x H),
so adjacent cells tile seamlessly without cliffs or gaps.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


# ===========================================================================
# Helper
# ===========================================================================

def _solid_box(x_center: float, y_center: float, top_z: float,
               x_size: float, y_size: float,
               base_z: float = -0.2) -> trimesh.Trimesh:
    """Create a solid box from `base_z` up to `top_z`.

    Args:
        x_center, y_center: Center of the box in x-y plane.
        top_z: Height of the top surface.
        x_size, y_size: Dimensions in x and y.
        base_z: Bottom of the box. Defaults to -0.2 (below ground).

    Returns:
        A trimesh box mesh.
    """
    box_h = top_z - base_z
    if box_h <= 0:
        box_h = 0.01  # degenerate safety
    center_z = (top_z + base_z) / 2.0
    return trimesh.creation.box(
        (x_size, y_size, box_h),
        trimesh.transformations.translation_matrix([x_center, y_center, center_z]),
    )


def _compute_stair_layout(W: float, flat_mid: float, step_width: float,
                           flat_start: float = 0.0):
    """Compute stair layout that exactly fills cell width W.

    Args:
        W: Total cell width (m).
        flat_mid: Flat plateau/valley length at center (m).
        step_width: Horizontal depth of each step (m).
        flat_start: Explicit flat length reserved at each end for safe robot
            spawning (m). Stair count is reduced to fit within the remaining
            space. Default 0.0 uses old behaviour (pad absorbs remainder).

    Returns:
        (num_steps, pad_each_side):
            num_steps  – stair steps per side.
            pad        – tiny residual flat length at each end (m), added to
                         flat_start so the total cell width is exactly W.
    """
    # Space available for two stair sections after reserving ends and centre
    available = W - flat_mid - 2.0 * flat_start
    per_side = max(0.0, available / 2.0)
    num_steps = max(1, int(per_side / step_width))
    actual_stair_len = num_steps * step_width
    # Tiny residual absorbed as additional pad
    remaining = available - 2 * actual_stair_len
    pad = remaining / 2.0
    return num_steps, pad


# ===========================================================================
# Training terrain: Trapezoid (UP then DOWN)
# ===========================================================================

def _resolve_step_height(difficulty: float, cfg) -> float:
    """Map difficulty to step height, respecting flat_threshold.

    If flat_threshold > 0:
      - difficulty < flat_threshold  → step_height = 0 (flat terrain, row 0)
      - difficulty >= flat_threshold → linearly interpolate step_height_range
                                       across the remaining difficulty range
    This lets row 0 be flat while rows 1-N span the full step_height_range.

    Example (num_rows=7, flat_threshold=1/6):
      Row 0 (difficulty=0/6): 0 cm   (flat)
      Row 1 (difficulty=1/6): 15 cm
      Row 6 (difficulty=6/6): 20 cm
    """
    lo, hi = cfg.step_height_range
    ft = cfg.flat_threshold
    if ft > 0:
        if difficulty < ft:
            return 0.0
        t = (difficulty - ft) / (1.0 - ft)
        return lo + t * (hi - lo)
    return lo + difficulty * (hi - lo)


def trapezoid_stairs_terrain(
    difficulty: float, cfg: "TrapezoidStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Trapezoid staircase: flat → UP → plateau → DOWN → flat.

    Side view:
        z=0 ──────────┐                          ┌────────── z=0
                       └── UP ──┐  plateau  ┌── DOWN ──┘
        | flat_start | stairs   |  flat_mid |  stairs  | flat_start |
        0                                                            W

    flat_start reserves explicit flat ground at each end for safe robot
    spawning. A full-cell ground floor at z=0 fills all gaps.
    """
    W, H = cfg.size
    step_height = _resolve_step_height(difficulty, cfg)
    sw = cfg.step_width
    flat_mid = cfg.flat_mid_length
    flat_start = cfg.flat_start_length

    num_steps, pad = _compute_stair_layout(W, flat_mid, sw, flat_start)
    total_height = num_steps * step_height
    # Effective flat zone at each end (reserved + tiny remainder)
    flat_end = flat_start + pad

    y_c = H / 2.0
    meshes = []

    # Ground floor: covers entire cell at z=0 (fills ALL gaps)
    meshes.append(_solid_box(W / 2, y_c, 0.0, W, H))

    if total_height > 0:
        # Ascending stairs (left side)
        x0 = flat_end
        for k in range(num_steps):
            top_z = (k + 1) * step_height
            meshes.append(_solid_box(x0 + (k + 0.5) * sw, y_c, top_z, sw, H))

        # Plateau
        plateau_x = flat_end + num_steps * sw
        meshes.append(_solid_box(plateau_x + flat_mid / 2, y_c, total_height, flat_mid, H))

        # Descending stairs (right side)
        x0_desc = plateau_x + flat_mid
        for k in range(num_steps):
            top_z = total_height - (k + 1) * step_height
            if top_z > 0:
                meshes.append(_solid_box(x0_desc + (k + 0.5) * sw, y_c, top_z, sw, H))

    # Origin at stair entry (x = flat_end).
    # Combined with negative x-offsets during spawn, robots always start
    # somewhere on flat ground with the first step directly ahead.
    origin = np.array([flat_end, y_c, 0.0])
    return meshes, origin


# ===========================================================================
# Training terrain: Inverted Trapezoid (DOWN then UP)
# ===========================================================================

def inverted_trapezoid_stairs_terrain(
    difficulty: float, cfg: "InvertedTrapezoidStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Inverted trapezoid: flat(z=0) → DOWN → valley(z<0) → UP → flat(z=0).

    Side view:
        z=0 ────┐                                     ┌──── z=0
                 └── DOWN ──┐  valley(z=-h) ┌── UP ──┘
        |  pad  |  stairs   |   flat_mid    | stairs  |  pad  |
        0                                                     W

    Both edges are at z=0 for seamless tiling. Valley goes below z=0.
    Left/right pads and each step are solid boxes covering the full cell
    width (H), ensuring no y-direction gaps.
    """
    W, H = cfg.size
    step_height = _resolve_step_height(difficulty, cfg)
    sw = cfg.step_width
    flat_mid = cfg.flat_mid_length
    flat_start = cfg.flat_start_length

    num_steps, pad = _compute_stair_layout(W, flat_mid, sw, flat_start)
    total_height = num_steps * step_height
    flat_end = flat_start + pad  # effective flat zone at each end

    y_c = H / 2.0
    # Base of all boxes: well below the deepest valley point
    deep_base = -(total_height + 0.5)
    meshes = []

    # Left flat zone at z=0 (solid from deep_base to z=0)
    if flat_end > 0.001:
        meshes.append(_solid_box(flat_end / 2, y_c, 0.0, flat_end, H, base_z=deep_base))

    if total_height > 0:
        # Descending stairs (left side, going down from z=0 to z=-total_height)
        x0 = flat_end
        for k in range(num_steps):
            top_z = -(k + 1) * step_height
            meshes.append(_solid_box(x0 + (k + 0.5) * sw, y_c, top_z, sw, H, base_z=deep_base))

        # Valley floor at z=-total_height
        valley_x = flat_end + num_steps * sw
        meshes.append(_solid_box(valley_x + flat_mid / 2, y_c, -total_height, flat_mid, H, base_z=deep_base))

        # Ascending stairs (right side, going up from z=-total_height to z=0)
        x0_asc = valley_x + flat_mid
        for k in range(num_steps):
            top_z = -total_height + (k + 1) * step_height
            meshes.append(_solid_box(x0_asc + (k + 0.5) * sw, y_c, top_z, sw, H, base_z=deep_base))
    else:
        # difficulty=0 → flat terrain
        meshes.append(_solid_box(W / 2, y_c, 0.0, W, H, base_z=deep_base))

    # Right flat zone at z=0
    right_flat_x = W - flat_end
    if flat_end > 0.001:
        meshes.append(_solid_box(right_flat_x + flat_end / 2, y_c, 0.0, flat_end, H, base_z=deep_base))

    # Origin at stair entry (x = flat_end).
    origin = np.array([flat_end, y_c, 0.0])
    return meshes, origin


# ===========================================================================
# Evaluation terrains: One-way straight stairs (kept for eval_stairs.py)
# ===========================================================================

def straight_ascending_stairs_terrain(
    difficulty: float, cfg: "StraightAscendingStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Straight ascending staircase in +x direction (evaluation only)."""
    W, H = cfg.size
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    sw = cfg.step_width
    stair_w = cfg.stair_width
    flat_start = cfg.flat_start_length
    flat_end = cfg.flat_end_length

    stair_section = W - flat_start - flat_end
    num_steps = max(1, int(stair_section / sw))
    total_height = num_steps * step_height
    y_c = H / 2.0

    meshes = []
    meshes.append(_solid_box(flat_start / 2, y_c, 0.0, flat_start, stair_w))
    for k in range(num_steps):
        top_z = (k + 1) * step_height
        meshes.append(_solid_box(flat_start + (k + 0.5) * sw, y_c, top_z, sw, stair_w))
    top_x = flat_start + num_steps * sw
    meshes.append(_solid_box(top_x + flat_end / 2, y_c, total_height, flat_end, stair_w))

    origin = np.array([flat_start / 2.0, y_c, 0.0])
    return meshes, origin


def straight_descending_stairs_terrain(
    difficulty: float, cfg: "StraightDescendingStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Straight descending staircase in +x direction (evaluation only)."""
    W, H = cfg.size
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    sw = cfg.step_width
    stair_w = cfg.stair_width
    flat_start = cfg.flat_start_length
    flat_end = cfg.flat_end_length

    stair_section = W - flat_start - flat_end
    num_steps = max(1, int(stair_section / sw))
    total_height = num_steps * step_height
    y_c = H / 2.0

    meshes = []
    meshes.append(_solid_box(flat_start / 2, y_c, total_height, flat_start, stair_w))
    for k in range(num_steps):
        top_z = total_height - (k + 1) * step_height
        if top_z < 0:
            top_z = 0.0
        meshes.append(_solid_box(flat_start + (k + 0.5) * sw, y_c, top_z, sw, stair_w))
    bottom_x = flat_start + num_steps * sw
    meshes.append(_solid_box(bottom_x + flat_end / 2, y_c, 0.0, flat_end, stair_w))

    origin = np.array([flat_start / 2.0, y_c, total_height])
    return meshes, origin


# ===========================================================================
# Configuration classes
# ===========================================================================

@configclass
class TrapezoidStairsTerrainCfg(SubTerrainBaseCfg):
    """Trapezoid: flat_start → UP → plateau → DOWN → flat_start.

    Both edges at z=0 for seamless tiling. Full cell coverage, no gaps.
    flat_start_length reserves explicit flat ground at each end for safe
    robot spawning, reducing stair count to fit within the remaining space.
    """

    function = trapezoid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """Min and max step height (m). Difficulty interpolates between these."""

    step_width: float = 0.28
    """Horizontal depth of each step (m)."""

    flat_mid_length: float = 1.0
    """Flat plateau at the top between ascending and descending sections (m)."""

    flat_start_length: float = 0.0
    """Flat ground reserved at each end of the cell (m) for safe spawning.
    The robot origin is placed at the centre of this zone.
    Stair count is reduced so that flat_start + stairs + flat_mid fits in W.
    Default 0.0 uses old behaviour (tiny pad absorbs rounding remainder)."""

    flat_threshold: float = 0.0
    """Difficulty threshold below which the terrain is flat (step_height=0).
    Rows with difficulty < flat_threshold produce flat ground.
    Rows with difficulty >= flat_threshold linearly span step_height_range.
    Set to 1/(num_rows-1) to make exactly row 0 flat.
    Default 0.0 disables this feature (standard linear interpolation)."""


@configclass
class InvertedTrapezoidStairsTerrainCfg(SubTerrainBaseCfg):
    """Inverted trapezoid: flat_start → DOWN → valley(z<0) → UP → flat_start.

    Both edges at z=0 for seamless tiling. Valley goes below ground level.
    """

    function = inverted_trapezoid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """Min and max step height (m). Difficulty interpolates between these."""

    step_width: float = 0.28
    """Horizontal depth of each step (m)."""

    flat_mid_length: float = 1.0
    """Flat valley floor between descending and ascending sections (m)."""

    flat_start_length: float = 0.0
    """Flat ground reserved at each end of the cell (m) for safe spawning.
    See TrapezoidStairsTerrainCfg.flat_start_length for details."""

    flat_threshold: float = 0.0
    """Difficulty threshold below which the terrain is flat (step_height=0).
    See TrapezoidStairsTerrainCfg.flat_threshold for details."""


# --- Evaluation-only terrain configs (one-way straight stairs) ---

@configclass
class StraightAscendingStairsTerrainCfg(SubTerrainBaseCfg):
    """Straight ascending staircase (evaluation only)."""

    function = straight_ascending_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    step_width: float = 0.28
    stair_width: float = 6.0
    flat_start_length: float = 1.5
    flat_end_length: float = 1.5


@configclass
class StraightDescendingStairsTerrainCfg(SubTerrainBaseCfg):
    """Straight descending staircase (evaluation only)."""

    function = straight_descending_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    step_width: float = 0.28
    stair_width: float = 6.0
    flat_start_length: float = 1.5
    flat_end_length: float = 1.5
