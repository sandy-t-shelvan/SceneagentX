"""Build blocks from the reproduction plan.

B0 and B1 are Phase 1. Later blocks are registered so status.json
has a row for each gate, and the pipeline can refuse them cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    id: str
    name: str
    phase: int
    summary: str


BLOCKS: tuple[Block, ...] = (
    Block("B0", "environment", 1, "GPU box, seven envs, status helper"),
    Block("B1", "capture", 1, "Sharp frames and AprilTag coverage"),
    Block("B2", "colmap", 2, "COLMAP poses for one orbit"),
    Block("B3", "splatfacto", 2, "3D Gaussian splat for one orbit"),
    Block("B4", "align", 2, "Scale and align orbits 2 and 3"),
    Block(
        "B5",
        "masks",
        3,
        "Grounded-SAM-2 masks. The LangSplat codebook is a separate unstarted item under block 2.",
    ),
    Block("B6", "lift", 3, "Lift masks onto Gaussians"),
    Block(
        "B7",
        "object vector for physics",
        3,
        "DINOv2 + OpenCLIP object vector for physics. This is not the LangSplat codebook.",
    ),
    Block("B8", "mesh", 4, "2DGS mesh and Chamfer / F1 score"),
    Block("B9", "parts", 4, "Cap, body, and hidden tip"),
    Block("B10", "physics", 5, "Per-part and per-Gaussian physics"),
    Block("B11", "articulate", 5, "URDF rest and pull test"),
    Block("B12", "package", 5, "USDZ package and score"),
    Block("B13", "procedural_sisters", 6, "Ten procedural sisters"),
    Block("B14", "generative_sisters", 6, "TRELLIS sisters"),
    Block("B15", "deformable", 6, "Soft-body drop test"),
    Block("B16", "agent_loop", 6, "Unattended rerun of the gates"),
    Block("B17", "second_object", 6, "Hinged object, no tool edits"),
)

PHASE1_IDS: tuple[str, ...] = tuple(block.id for block in BLOCKS if block.phase == 1)

_BY_ID = {block.id: block for block in BLOCKS}


def get_block(block_id: str) -> Block:
    key = block_id.upper()
    if key not in _BY_ID:
        known = ", ".join(_BY_ID)
        raise KeyError(f"Unknown block {block_id!r}. Known blocks: {known}")
    return _BY_ID[key]


def blocks_through(last_id: str) -> tuple[Block, ...]:
    last = get_block(last_id)
    ordered = []
    for block in BLOCKS:
        ordered.append(block)
        if block.id == last.id:
            return tuple(ordered)
    return tuple(ordered)
