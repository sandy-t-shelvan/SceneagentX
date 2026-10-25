"""Run can orbits 1–3 from Block B files through D, then E, then F.

Block C is not executed here. The C line is copied from ingest/block_c.json
when that file has the orbit. A missing file stays skipped and is not a pass.
SAM, COLMAP, and splatfacto are not rerun. Outputs go to data/can/chain/orbitN/.
This module does not call publish_can or refresh the shared data/can scores.
A skipped or not_scheduled check is never a pass. learned_prior stays false.
M4 stays unmet without a scale weight. A missing checkpoint, a marker-cap
check, and an Isaac drop are not scheduled for this video and do not fail the
bake. Block D passes when the Block C object ply, the prior rows, and the
physics bake match.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from tools.block_d import (
    apply_scene_mean,
    fuse_physics,
    DROP_NOT_SCHEDULED,
    reduce_prior,
    run_learned_prior,
    start_d2,
    write_gaussian_physics,
    write_physics_material_usda,
    write_prior_input,
    write_scene_mean_labels,
)
from tools.block_e import (
    collision_hulls,
    extract_object_mesh,
    record_joint_sweep,
    resolve_parts,
    route_deformable,
    write_deformable,
)
from tools.block_f import align_similarity, camera_intrinsics, inspect_scene, orient_background, write_scene_usdz
from tools.block_gates import GateError
from tools.can_block_b_check import check_orbit, probe_b123
from tools.orbit_blocks import assign_feature_indices, resolve_scene_mean

ORBITS = ("orbit1", "orbit2", "orbit3")
HEADER = (
    "# Can orbit chain\n"
    "\n"
    "Object: can. Orbits 1–3, one orbit at a time, starting from Block B files already on disk. "
    "SAM, COLMAP, and splatfacto are not rerun. "
    "Block C is read from ingest/block_c.json. A missing file stays skipped and is not a pass. "
    "Block D passes when the Block C object ply and the physics bake match. "
    "M4 stays unmet without weighed objects. learned_prior stays false.\n"
    "\n"
)
_CHECKPOINT_REASON = "checkpoint is a project weight and is not produced by this video."
_M4_REASON = "a vision mass is an unchecked estimate."
_M4B_REASON = "M4b is a marker-cap breakaway and is not scheduled for this object."
_D2_FRAMES = (
    ("iou", "orbit1", "frame_00167.png"),
    ("iou", "orbit2", "frame_00001.png"),
    ("iou", "orbit3", "frame_00125.png"),
    ("reviews", "frame_00001.png"),
)
_REVIEW_STEMS = {"orbit1": "frame_00167", "orbit2": "frame_00001", "orbit3": "frame_00125"}
_PROTECTED = (
    "physics.json",
    "asset.urdf",
    "asset.usda",
    "deformable.json",
    "orientation.json",
    "mesh_score.json",
    "mesh/mesh.ply",
    "background.ply",
    "ingest/block_a.json",
    "ingest/block_b.json",
    "ingest/block_b_b1.json",
    "ingest/block_b_b2.json",
    "ingest/block_b_b3.json",
    "ingest/block_b_b4.json",
    "ingest/block_b_orbit1.json",
    "ingest/block_b_orbit2.json",
    "ingest/block_b_orbit3.json",
    "ingest/block_d.json",
    "ingest/block_e.json",
    "ingest/block_f.json",
    "ingest/hulls.json",
    "ingest/joint_sweep.json",
    "codebook.json",
    "codebook_orbit2.json",
    "codebook_orbit3.json",
    "masks/orbit1/object.json",
    "masks/orbit1/gaussian_in_mask.npy",
    "masks/orbit2/object.json",
    "masks/orbit2/gaussian_in_mask.npy",
    "masks/orbit3/object.json",
    "masks/orbit3/gaussian_in_mask.npy",
    "semantics/gaussian_feature_idx.npy",
    "semantics/orbit1/instances.json",
    "semantics/orbit2/instances.json",
    "semantics/orbit3/instances.json",
)


def run_chain(
    can: Path,
    markdown: Path | None = None,
    orbits: tuple[str, ...] = ORBITS,
) -> dict[str, Any]:
    """Run orbits 1–3. Shared can physics, mesh, and asset files are restored if touched."""
    can = _require_can(can)
    for orbit in orbits:
        _orbit(orbit)
    md = Path(markdown) if markdown is not None else can / "ingest" / "orbit_chain.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(HEADER, encoding="utf-8")
    snap = _snapshot(can)
    results = []
    try:
        for orbit in orbits:
            results.append(run_orbit(can, orbit, md))
    finally:
        _guard_shared(can, snap)
    summary = {
        "object": "can",
        "markdown": str(md),
        "orbits": {item["orbit"]: _brief_orbit(item) for item in results},
    }
    dest = can / "chain" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return {"markdown": md, "orbits": results, "summary": summary}


def run_orbit(can: Path, orbit: str, markdown: Path) -> dict[str, Any]:
    """Measure B, read C from ingest/block_c.json, then run D, E, and F."""
    can = _require_can(can)
    orbit = _orbit(orbit)
    markdown = Path(markdown)
    if not markdown.is_file():
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(HEADER, encoding="utf-8")
    snap = _snapshot(can)
    chain = _fresh_chain(can, orbit)
    try:
        with markdown.open("a", encoding="utf-8") as handle:
            handle.write(f"## {orbit}\n\n")
        measured = run_block_b(can, orbit)
        b_path = _write_record(chain / "ingest" / "block_b.json", measured["record"])
        _append_rendered(markdown, b_path)
        recorded = _c_record(can, orbit)
        c_path = _write_record(chain / "ingest" / "block_c.json", recorded)
        _append_rendered(markdown, c_path)
        d_result = run_block_d(can, orbit, measured["inputs"], chain)
        _append_rendered(markdown, d_result["record_path"])
        e_result = run_block_e(can, orbit, d_result["output"])
        _append_rendered(markdown, e_result["record_path"])
        f_result = run_block_f(can, orbit, e_result["output"])
        _append_rendered(markdown, f_result["record_path"])
    finally:
        _guard_shared(can, snap)
    return {
        "orbit": orbit,
        "chain": chain,
        "b_inputs": measured["inputs"],
        "b": measured["record"],
        "c": recorded,
        "d_inputs": d_result["inputs"],
        "d_output": d_result["output"],
        "d": d_result["record"],
        "e_input": e_result["input"],
        "e_output": e_result["output"],
        "e": e_result["record"],
        "f_input": f_result["input"],
        "f_output": f_result["output"],
        "f": f_result["record"],
    }


def run_block_b(can: Path, orbit: str) -> dict[str, Any]:
    """Re-measure Block B from files already on disk. Does not train."""
    can = _require_can(can)
    orbit = _orbit(orbit)
    check_orbit(can, orbit)
    probe_b123(can, orbit)
    check = _load(can / "debug" / f"block_b_{orbit}_check.json")
    probe = _load(can / "debug" / f"block_b_{orbit}_b123_probe.json")
    score = _load(can / "ingest" / f"block_b_{orbit}.json")
    inputs = _b_inputs(can, orbit)
    steps = check.get("steps") if isinstance(check.get("steps"), dict) else {}
    parts = [
        _b_part("B1", score, steps.get("B1") or {}, probe.get("B1") or {}),
        _b_part("B2", score, steps.get("B2") or {}, probe.get("B2") or {}),
        _b_part("B3", score, steps.get("B3") or {}, probe.get("B3") or {}),
        _b_part("B4", score, steps.get("B4") or {}, probe.get("B4") or {}),
        _milestone_m1(score, check, steps.get("A") or {}),
        _milestone_m2(score, check, steps.get("B4") or {}),
    ]
    record = _block_record("B", orbit, parts, state=score.get("state") or _roll_state(parts))
    record["learned_prior"] = False
    return {"inputs": inputs, "record": record}


def run_block_d(
    can: Path,
    orbit: str,
    b_inputs: dict[str, Path],
    chain: Path,
) -> dict[str, Any]:
    """Run D1.2–D3 on this orbit's Block B files. Writes stay inside the chain folder."""
    can = _require_can(can)
    orbit = _orbit(orbit)
    chain = _require_chain(can, chain)
    inputs = {key: Path(value) for key, value in b_inputs.items()}
    mask_path = inputs["mask"]
    expected_mask = can / "masks" / orbit / "gaussian_in_mask.npy"
    if mask_path.resolve() != expected_mask.resolve():
        raise RuntimeError("Block D must read this orbit's Block B mask")
    codebook_path = inputs["codebook"]
    splat = inputs["splat"]
    codebook = _codebook_for_d12(can, orbit, _load(codebook_path))
    obj = _load(inputs["object"])
    inside = np.load(mask_path) if mask_path.is_file() else None
    d12 = _run_d12(chain, inside, codebook)
    _note_splat_and_ply(d12["part"], can, orbit)
    prior_status, d2_status = _object_dir_writers(
        chain,
        splat,
        d12.get("assigned"),
        _d2_frames(can),
        _bbox(obj),
    )
    d14 = _run_d14(can, chain, inside)
    d15 = _run_d15(chain, inside, d14.get("p_i"), checkpoint_present=d14.get("checkpoint_present") is True)
    d3 = _run_d3(chain, d2_status, mask_path)
    physics = _load(chain / "physics.json") if (chain / "physics.json").is_file() else {}
    labeled = d12["part"]["numbers"].get("n_object")
    parts = [
        _d11_part(codebook, labeled if isinstance(labeled, int) else None),
        d12["part"],
        _d13_part(chain, _mark_ply_match(can, orbit, chain, prior_status)),
        d14["part"],
        d15,
        _d2_part(chain, d2_status),
        d3,
        _m4_part(physics),
        _m4b_part(),
    ]
    record = _finish_d_record(can, orbit, chain, parts, d2_status, obj.get("covers_room") is True)
    path = _write_record(chain / "ingest" / "block_d.json", record)
    return {
        "inputs": inputs,
        "output": chain,
        "record": _load(path),
        "record_path": path,
        "d2_status": d2_status,
    }


def _block_d_state(chain: Path) -> str:
    """Partial only when this orbit's physics.json has material and mass_kg.

    A measured D1.1 or D1.2 does not start the block. Matches tools.sceneagent_blocks._block_d.
    """
    from tools.sceneagent_blocks import _block_d

    scored = _block_d("can", chain, None)
    state = scored.get("state")
    if state not in {"not_started", "partial"}:
        raise GateError("Block D state must come from physics.json")
    return str(state)


def run_block_e(can: Path, orbit: str, d_output: Path) -> dict[str, Any]:
    """Run E on D's folder using this orbit's Block B mask. Does not call publish_can."""
    can = _require_can(can)
    orbit = _orbit(orbit)
    folder = _require_chain(can, d_output)
    mask = can / "masks" / orbit / "gaussian_in_mask.npy"
    splat = can / "colmap" / orbit / "splat.ply"
    mesh_dest = folder / "mesh" / "mesh.ply"
    mesh_status = extract_object_mesh(splat, mask, mesh_dest)
    reference = mesh_status.pop("reference_xyz", None)
    if mesh_dest.is_file() and splat.is_file() and reference is not None:
        from tools.block_e import metres_per_unit_for, score_mesh

        block = score_mesh(mesh_dest, reference, metres_per_unit_for(folder))
        if block.get("achieved") is not True:
            block["achieved"] = False
        _dump(folder / "mesh_score.json", {"chamfer": block, "achieved": block.get("achieved") is True})
    parts_list = resolve_parts(None, None)
    _dump(folder / "ingest" / "parts.json", {"parts": parts_list, "learned_prior": False})
    hull_target = mesh_dest if mesh_dest.is_file() else folder / "mesh" / "missing.ply"
    hull = collision_hulls(hull_target)
    _dump(folder / "ingest" / "hulls.json", hull)
    sweep_path = folder / "ingest" / "joint_sweep.json"
    existing = _load(sweep_path)
    if existing.get("joint_sweep") == "ran" and _sweep_has_numbers(existing):
        sweep = existing
    else:
        sweep = record_joint_sweep(None)
        _dump(sweep_path, sweep)
    physics = _load(folder / "physics.json")
    kind = physics.get("deformable_type") if isinstance(physics.get("deformable_type"), str) else None
    if physics and kind:
        route = route_deformable(kind)
        deformable = write_deformable(folder)
        if deformable.get("route") != route:
            deformable["route"] = route
            deformable["learned_prior"] = False
            _dump(folder / "deformable.json", deformable)
    else:
        route = None
        stale = folder / "deformable.json"
        if stale.is_file():
            leftover = _load(stale)
            if leftover.get("cloth") is True or leftover.get("rope") is True or leftover.get("soft") is True:
                raise GateError("a missing physics record must not invent cloth, rope, or soft")
            stale.unlink()
    parts = [
        _e1_part(mesh_dest, mesh_status, splat, mask),
        _e2_part(folder, parts_list, hull),
        _e3_part(folder, sweep),
        _e4_part(folder, route),
        _unmeasured("M5", "M5 needs a radio, named moving parts, and an Isaac joint sweep."),
        _unmeasured("M5b", "M5b needs a towel and a hose. No cloth, rope, or soft body was built."),
    ]
    record = _block_record("E", orbit, parts, state=_e_block_state(folder))
    path = _write_record(folder / "ingest" / "block_e.json", record)
    return {"input": folder, "output": folder, "record": _load(path), "record_path": path}


def run_block_f(can: Path, orbit: str, e_output: Path) -> dict[str, Any]:
    """Run F on E's folder. Missing world_T_object, from skipping C, is not alignment."""
    can = _require_can(can)
    orbit = _orbit(orbit)
    folder = _require_chain(can, e_output)
    mesh = folder / "mesh" / "mesh.ply"
    copy_c1_pose(can, orbit, folder)
    world = _first_file(folder, ("world_T_object.json", "ingest/world_T_object.json", "objects/can/world_T_object.json"))
    stem = _REVIEW_STEMS[orbit]
    capture = can / "reviews" / f"{stem}.png"
    mask = can / "reviews" / f"{stem}_mask.png"
    intrinsics = _orbit_intrinsics(can, orbit)
    mesh_path = mesh if mesh.is_file() else None
    capture_path = capture if capture.is_file() else None
    mask_path = mask if mask.is_file() else None
    gaps = _f1_gaps(mesh_path, capture_path, mask_path, world, intrinsics)
    align = align_similarity(
        mesh_path,
        capture_path,
        mask_path,
        world,
        intrinsics,
        metres_per_unit=None,
        dest=folder / "align.json",
    )
    if world is None and (folder / "align.json").is_file() and _load(folder / "align.json").get("achieved") is True:
        raise GateError("alignment cannot be achieved without world_T_object")
    background = _orbit_background(can, orbit, folder)
    orient = None
    if background is not None:
        orient = orient_background(background, None, dest=folder / "orientation.json")
    mesh_written = mesh.is_file() and mesh.stat().st_size > 0
    asset = _first_file(folder, ("asset.usd", "asset.usda", "asset.usdc"))
    physics = _load(folder / "physics.json")
    usdz = None
    if mesh_written and asset is not None and physics:
        from tools.block_f import compose_scene_usda

        compose_scene_usda(
            folder / "scene.usda",
            asset_usd=asset,
            physics=physics,
            cameras=[],
            background=background,
            plane=None,
            franka_usd=None,
        )
        usdz = write_scene_usdz(
            folder / "scene.usdz",
            usda_path=folder / "scene.usda",
            asset_usd=asset,
            physics=physics,
        )
    report = inspect_scene(
        scene=folder / "scene.usdz" if usdz else None,
        cameras=None,
        telemetry=None,
        folder=folder,
        write_report=True,
    )
    written_report = _load(folder / "inspection_report.json")
    if written_report.get("settle_within_2s") is True or report.get("settle_within_2s") is True:
        raise GateError("settle_within_2s cannot be true without Isaac telemetry")
    parts = [
        _f1_part(folder, align, gaps),
        _f2_part(folder, orient, background is not None),
        _f3_part(folder, mesh_written, asset is not None, bool(physics)),
        _f4_part(folder, written_report),
        _m6_part(folder),
        _m7_part(folder, written_report),
        _m0_part(folder),
    ]
    record = _block_record("F", orbit, parts, state=_f_block_state(folder, parts))
    path = _write_record(folder / "ingest" / "block_f.json", record)
    return {"input": folder, "output": folder, "record": _load(path), "record_path": path}


def render_block_file(path: Path) -> str:
    """Render a block JSON. achieved: true is copied only when the file says true."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise GateError("block record must be an object")
    return render_block(record)


def render_block(record: dict[str, Any]) -> str:
    """One block section. A false achieved flag is not rendered as achieved."""
    lines = [
        f"### {record.get('block')}",
        "",
        f"Block {record.get('block')} state `{record.get('state')}` {_achieved_cell(record.get('achieved'))}.",
        "",
        "| step | state | achieved | numbers |",
        "| --- | --- | --- | --- |",
    ]
    for part in record.get("parts") or []:
        if not isinstance(part, dict):
            continue
        cells = (
            _cell(str(part.get("id"))),
            _cell(str(part.get("state"))),
            _achieved_cell(part.get("achieved")),
            _cell(_numbers_text(part)),
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the can orbit chain from Block B through F")
    parser.add_argument("--can", type=Path, default=Path("data/can"))
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_chain(args.can, args.markdown)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


def _saved_scene_mean(can: Path, orbit: str, vector: Any) -> list[float] | None:
    """Return a stored scene-mean vector that is not a copy of the object vector."""
    if not isinstance(vector, list) or not vector:
        return None
    obj = np.asarray(vector, dtype=float).reshape(-1)
    paths = [
        can / f"codebook_gaussians_{orbit}.npz",
        can / "semantics" / orbit / "features.codebook.npz",
        can / "semantics" / "features.physics_vector.npz",
    ]
    if orbit == "orbit1":
        paths.insert(0, can / "codebook_gaussians.npz")
    for path in paths:
        if not path.is_file():
            continue
        try:
            with np.load(path) as book:
                if "scene_mean" not in book.files:
                    continue
                scene = np.asarray(book["scene_mean"], dtype=float)
        except (OSError, ValueError):
            continue
        if scene.ndim != 1 or scene.size == 0 or not np.isfinite(scene).all():
            continue
        if scene.shape != obj.shape or bool(np.allclose(scene, obj)):
            continue
        return [float(value) for value in scene]
    return None


def _codebook_for_d12(can: Path, orbit: str, codebook: dict[str, Any]) -> dict[str, Any]:
    """Use a scene mean already stored on disk. Do not invent one or copy the object vector."""
    if resolve_scene_mean(codebook).get("measured") is True:
        return codebook
    found = _saved_scene_mean(can, orbit, codebook.get("vector"))
    if found is None:
        return codebook
    merged = dict(codebook)
    merged["scene_mean"] = found
    merged["scene_mean_measured"] = True
    if resolve_scene_mean(merged).get("measured") is not True:
        return codebook
    return merged


def _note_splat_and_ply(part: dict[str, Any], can: Path, orbit: str) -> None:
    """Record the full-splat index and the Block C ply as different counts."""
    from tools.step02_splatfacto import count_gaussians

    ply = _block_c_object_ply(can, orbit)
    ply_n = count_gaussians(ply) if ply.is_file() else None
    numbers = part["numbers"]
    n = numbers.get("n")
    n_object = numbers.get("n_object")
    if isinstance(n, int):
        numbers["n_splat"] = n
    if isinstance(ply_n, int):
        numbers["ply_vertices"] = ply_n
    if not isinstance(ply_n, int) or not isinstance(n, int):
        return
    if n == ply_n and n_object == ply_n:
        return
    note = (
        f" The index labels the full splat ({n} Gaussians, {n_object} object labels). "
        f"The Block C object ply has {ply_n} vertices. The bake uses the ply."
    )
    reason = str(part.get("reason") or "")
    if "Block C object ply has" not in reason:
        part["reason"] = reason + note


def _mark_ply_match(can: Path, orbit: str, chain: Path, status: dict[str, Any]) -> dict[str, Any]:
    from tools.step02_splatfacto import count_gaussians

    marked = dict(status)
    ply = _block_c_object_ply(can, orbit)
    ply_n = count_gaussians(ply) if ply.is_file() else None
    prior = chain / "semantics" / "prior_input.npy"
    prior_n = int(np.load(prior, mmap_mode="r").shape[0]) if prior.is_file() else None
    marked["ply_vertices"] = ply_n
    marked["matches_object_ply"] = ply_n is not None and prior_n == ply_n
    return marked


def _weighed(physics: dict[str, Any]) -> bool:
    if physics.get("weighed_kg") is not None:
        return True
    check = physics.get("mass_check")
    return isinstance(check, dict) and check.get("checked") is True


def _m4_part(physics: dict[str, Any]) -> dict[str, Any]:
    if not _weighed(physics):
        return _step(
            "M4",
            "not_scheduled",
            False,
            {"achieved": False, "m4_met": False, "mass_checked": False, "learned_prior": False},
            _M4_REASON,
        )
    return _step(
        "M4",
        "unmeasured",
        False,
        {
            "achieved": False,
            "m4_met": False,
            "mass_checked": True,
            "weighed_kg": physics.get("weighed_kg"),
            "learned_prior": False,
        },
        "A scale weight is present. M4 still needs the weighed set, mass error, material accuracy, and settle telemetry.",
    )


def _m4b_part() -> dict[str, Any]:
    return _step(
        "M4b",
        "not_scheduled",
        False,
        {"achieved": False, "learned_prior": False},
        _M4B_REASON,
    )


def _schedule_drop(path: Path) -> None:
    """Replace a hang stamp with the video schedule. Do not invent telemetry or change the mass."""
    physics = _load(path)
    if not physics or physics.get("drop_test") == "ran":
        return
    if physics.get("settle_within_2s") is True:
        raise GateError("settle_within_2s cannot be true without telemetry")
    if physics.get("learned_prior") is True:
        raise GateError("learned_prior cannot be true")
    updated = dict(physics)
    updated["drop_test"] = "not_run"
    updated["drop_test_reason"] = DROP_NOT_SCHEDULED
    if "settle_within_2s" in updated and updated.get("settle_within_2s") is not None:
        updated["settle_within_2s"] = None
    if updated != physics:
        _dump(path, updated)


def _run_d12(chain: Path, inside: np.ndarray | None, codebook: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_scene_mean(codebook) if codebook else {"measured": False, "reason": "no codebook vector"}
    vector = codebook.get("vector")
    scene = codebook.get("scene_mean")
    separate = resolved.get("measured") is True and isinstance(vector, list) and isinstance(scene, list)
    _dbg200(
        "E",
        "tools/orbit_chain.py:_run_d12",
        "d12 branch",
        {
            "chain": chain.name,
            "branch": "separate" if inside is not None and separate else ("vector_only" if inside is not None and isinstance(vector, list) else "unstarted"),
            "scene_mean": isinstance(scene, list),
            "scene_mean_measured": codebook.get("scene_mean_measured") is True,
            "resolved": resolved.get("measured") is True,
        },
    )
    assigned = None
    if inside is not None and separate:
        obj = np.asarray(vector, dtype=float)
        scene_vec = np.asarray(scene, dtype=float)
        mask = np.asarray(inside)
        applied = apply_scene_mean(mask, obj, scene_vec)
        write_scene_mean_labels(chain, mask, obj, scene_vec)
        assigned = applied["assigned"]
        measured = applied.get("scene_mean_measured") is True
        state = "measured" if measured else "partial"
        reason = str(applied.get("reason") or "")
    elif inside is not None and isinstance(vector, list):
        index = assign_feature_indices(np.asarray(inside), False)
        dest = chain / "semantics" / "gaussian_feature_idx.npy"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, index)
        state = "partial"
        reason = str(resolved.get("reason") or "outside labels left at -1; no separate scene mean")
        measured = False
    else:
        state = "unstarted"
        reason = "mask or object vector is missing; outside labels were not invented"
        measured = False
    numbers: dict[str, Any] = {"scene_mean_measured": measured, "learned_prior": False}
    index_path = chain / "semantics" / "gaussian_feature_idx.npy"
    if index_path.is_file():
        index = np.load(index_path)
        numbers["n"] = int(index.size)
        numbers["n_object"] = int((index == 0).sum())
        numbers["n_scene"] = int((index == 1).sum())
        numbers["n_unassigned"] = int((index == -1).sum())
    return {
        "assigned": assigned,
        "part": _step("D1.2", state, False, numbers, reason),
    }


def _object_dir_writers(
    chain: Path,
    splat: Path,
    vectors: np.ndarray | None,
    frames: list[Path],
    bbox: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Write the prior inside the chain folder and reuse an existing vision physics.json.

    start_d2 is not called. A missing physics.json stays missing. An existing file is not deleted.
    """
    del frames, bbox
    root = chain / "_writer_root"
    produced = root / "data" / "can"
    produced.mkdir(parents=True, exist_ok=True)
    _require_inside(produced, chain)
    prior_status: dict[str, Any] = {"state": "unstarted", "file": None}
    try:
        prior_status = write_prior_input("can", root, ply=splat, vectors=vectors)
        _copy_produced(produced, chain)
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    if prior_status.get("state") != "written":
        stale_prior = chain / "semantics" / "prior_input.npy"
        if stale_prior.is_file() and not splat.is_file():
            stale_prior.unlink()
    d2_status = "reused" if (chain / "physics.json").is_file() else "missing"
    return prior_status, d2_status


def _block_c_object_ply(can: Path, orbit: str) -> Path:
    return can / "objects" / "can" / orbit / "object.ply"


def _score_block_c_bake(can: Path, chain: Path, orbit: str) -> dict[str, Any]:
    """Relaxed Block D pass: the bake matches the Block C object ply.

    A missing learned-prior checkpoint, an unweighed mass, and an unrun Isaac
    settle stay recorded and do not fail this bake. M4 stays unmet.
    """
    from tools.step02_splatfacto import count_gaussians

    ply = _block_c_object_ply(can, orbit)
    ply_n = count_gaussians(ply) if ply.is_file() else None
    prior_path = chain / "semantics" / "prior_input.npy"
    prior_n = int(np.load(prior_path, mmap_mode="r").shape[0]) if prior_path.is_file() else None
    physics = _load(chain / "physics.json") if (chain / "physics.json").is_file() else {}
    mass = _finite(physics.get("mass_kg"))
    material = physics.get("material")
    material_ok = isinstance(material, str) and bool(material.strip())
    row = np.array(
        [0.0 if _finite(physics.get(name)) is None else _finite(physics.get(name)) for name in ("mass_kg", "static_friction", "rigidity")],
        dtype=np.float64,
    )
    gauss_path = chain / "gaussian_physics.npy"
    gauss_n = None
    gauss_ok = False
    if gauss_path.is_file() and prior_n:
        field = np.load(gauss_path)
        gauss_n = int(field.shape[0]) if field.ndim == 2 else 0
        gauss_ok = field.ndim == 2 and field.shape[1] == 3 and gauss_n == prior_n and bool(np.allclose(field, row))
    usda = chain / "physics_material.usda"
    usda_ok = usda.is_file() and mass is not None and f"physics:mass = {mass}" in usda.read_text(encoding="utf-8")
    rows_ok = ply_n is not None and prior_n == ply_n
    met = bool(rows_ok and material_ok and mass is not None and gauss_ok and usda_ok)
    return {
        "met": met,
        "rule": "block_c_object_bake",
        "m4_met": False,
        "learned_prior": False,
        "ply_vertices": ply_n,
        "prior_rows": prior_n,
        "gauss_rows": gauss_n,
        "material": material_ok,
        "mass_kg": mass,
        "mass_checked": (physics.get("mass_check") or {}).get("checked") is True,
        "usda": usda_ok,
        "gaussian_physics": gauss_ok,
        "settle_within_2s": None,
    }


def _apply_relaxed_bake(parts: list[dict[str, Any]], bake: dict[str, Any]) -> None:
    if bake.get("met") is not True:
        return
    note = " Relaxed bake pass: Block C object ply rows match. Mass was not weighed and Isaac settle was not run."
    for part in parts:
        if part.get("id") in {"D1.3", "D2", "D3"}:
            part["achieved"] = True
            part["reason"] = str(part.get("reason") or "") + note


def _finish_d_record(
    can: Path,
    orbit: str,
    chain: Path,
    parts: list[dict[str, Any]],
    d2_status: str,
    covers_room: bool,
) -> dict[str, Any]:
    bake = _score_block_c_bake(can, chain, orbit)
    _apply_relaxed_bake(parts, bake)
    record = _block_record("D", orbit, parts, state=_block_d_state(chain))
    record["achieved"] = bake["met"] is True
    record["m4_met"] = False
    record["learned_prior"] = False
    record["d2_status"] = d2_status
    record["bake"] = bake
    if covers_room:
        record["covers_room"] = True
    return record


def _dbg200(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "200a3a",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(__import__("time").time() * 1000),
        }
        path = Path(__file__).resolve().parents[1] / "debug-200a3a.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion


def run_block_d_from_block_c(can: Path | None = None, orbits: tuple[str, ...] = ORBITS) -> list[dict[str, Any]]:
    """Bake Block D from each orbit's Block C object ply. Does not call the model or reset the chain."""
    can = _require_can(Path(can) if can is not None else Path("data/can"))
    saved: list[dict[str, Any]] = []
    for orbit in orbits:
        orbit = _orbit(orbit)
        chain = _require_chain(can, can / "chain" / orbit)
        codebook = _codebook_for_d12(
            can,
            orbit,
            _load(can / ("codebook.json" if orbit == "orbit1" else f"codebook_{orbit}.json")),
        )
        mask_path = can / "masks" / orbit / "gaussian_in_mask.npy"
        inside = np.load(mask_path) if mask_path.is_file() else None
        d12 = _run_d12(chain, inside, codebook)
        _note_splat_and_ply(d12["part"], can, orbit)
        d14 = _run_d14(can, chain, inside)
        d15 = _run_d15(chain, inside, d14.get("p_i"), checkpoint_present=d14.get("checkpoint_present") is True)
        status = "reused" if (chain / "physics.json").is_file() else "missing"
        d3 = _run_d3(chain, status, mask_path)
        physics = _load(chain / "physics.json") if (chain / "physics.json").is_file() else {}
        _dbg200(
            "C",
            "tools/orbit_chain.py:run_block_d_from_block_c",
            "m4 appended unmeasured",
            {
                "chain": chain.name,
                "mass_checked": (physics.get("mass_check") or {}).get("checked") is True,
                "weighed_present": physics.get("weighed_kg") is not None,
                "m4_caller": "not_scheduled",
                "m4b_caller": "not_scheduled",
            },
        )
        prior_state = "written" if (chain / "semantics" / "prior_input.npy").is_file() else "unstarted"
        labeled = d12["part"]["numbers"].get("n_object")
        parts = [
            _d11_part(codebook, labeled if isinstance(labeled, int) else None),
            d12["part"],
            _d13_part(chain, _mark_ply_match(can, orbit, chain, {"state": prior_state})),
            d14["part"],
            d15,
            _d2_part(chain, status),
            d3,
            _m4_part(physics),
            _m4b_part(),
        ]
        record = _finish_d_record(can, orbit, chain, parts, status, False)
        path = _write_record(chain / "ingest" / "block_d.json", record)
        written = _load(path)
        by_id = {part["id"]: part for part in written.get("parts") or []}
        _dbg200(
            "F",
            "tools/orbit_chain.py:run_block_d_from_block_c",
            "post-fix schedule",
            {
                "chain": chain.name,
                "block_achieved": written.get("achieved") is True,
                "m4_met": written.get("m4_met") is True,
                "learned_prior": written.get("learned_prior") is True,
                "n_copied": (by_id.get("D1.1") or {}).get("numbers", {}).get("n_copied"),
                "d2_reason": (by_id.get("D2") or {}).get("reason"),
                "d13_reason": (by_id.get("D1.3") or {}).get("reason"),
                "drop_test_reason": ((by_id.get("D3") or {}).get("numbers") or {}).get("drop_test_reason"),
                "states": {
                    part["id"]: {"state": part.get("state"), "achieved": part.get("achieved") is True}
                    for part in written.get("parts") or []
                    if isinstance(part, dict)
                },
            },
            run_id="post-fix",
        )
        saved.append(written)
    return saved


def _run_d14(can: Path, chain: Path, inside: np.ndarray | None) -> dict[str, Any]:
    del inside
    prior = chain / "semantics" / "prior_input.npy"
    checkpoint = _checkpoint(can, chain)
    prior_exists = prior.is_file()
    _dbg200(
        "A",
        "tools/orbit_chain.py:_run_d14",
        "d14 checkpoint",
        {
            "chain": chain.name,
            "prior_exists": prior_exists,
            "checkpoint": checkpoint is not None,
            "gate": "missing checkpoint" if checkpoint is None else "checkpoint present",
        },
    )
    if checkpoint is None:
        return {
            "part": _step(
                "D1.4",
                "not_scheduled",
                False,
                {"learned_prior": False, "checkpoint": False, "prior_input": prior_exists},
                _CHECKPOINT_REASON,
            ),
            "p_i": None,
            "checkpoint_present": False,
        }
    if not prior_exists:
        return {
            "part": _step(
                "D1.4",
                "unstarted",
                False,
                {"learned_prior": False, "checkpoint": True, "prior_input": False},
                "No prior_input.npy. The forward was not run.",
            ),
            "p_i": None,
            "checkpoint_present": True,
        }
    try:
        ran = run_learned_prior(np.load(prior), checkpoint)
    except GateError as exc:
        return {
            "part": _step(
                "D1.4",
                "unstarted",
                False,
                {"learned_prior": False, "checkpoint": True, "raised": True},
                str(exc),
            ),
            "p_i": None,
            "checkpoint_present": True,
        }
    if ran.get("learned_prior") is True:
        raise GateError("learned_prior cannot be true")
    return {
        "part": _step(
            "D1.4",
            "measured",
            False,
            {"learned_prior": False, "checkpoint": True, "n": ran.get("n")},
            "Checkpoint forward ran. learned_prior stays false.",
        ),
        "p_i": ran.get("p_i"),
        "checkpoint_present": True,
    }


def _run_d15(
    chain: Path,
    inside: np.ndarray | None,
    p_i: Any,
    checkpoint_present: bool = False,
) -> dict[str, Any]:
    dest = chain / "semantics" / "prior_reduced.json"
    _dbg200(
        "B",
        "tools/orbit_chain.py:_run_d15",
        "d15 gate",
        {
            "chain": chain.name,
            "p_i": isinstance(p_i, dict),
            "inside": inside is not None,
            "gate": "no p_i" if not isinstance(p_i, dict) or inside is None else "reduce",
        },
    )
    if not checkpoint_present:
        if dest.is_file():
            dest.unlink()
        return _step(
            "D1.5",
            "not_scheduled",
            False,
            {"learned_prior": False, "reduced": False},
            _CHECKPOINT_REASON,
        )
    if not isinstance(p_i, dict) or inside is None:
        if dest.is_file():
            dest.unlink()
        return _step(
            "D1.5",
            "unstarted",
            False,
            {"learned_prior": False, "reduced": False},
            "No p_i to reduce. An untrained prior was not reduced.",
        )
    reduced = reduce_prior(p_i, np.asarray(inside))
    if reduced.get("learned_prior") is True:
        raise GateError("learned_prior cannot be true")
    payload = {
        "object": reduced.get("object"),
        "parts": reduced.get("parts"),
        "learned_prior": False,
        "achieved": False,
    }
    _dump(dest, payload)
    saved = _load(dest)
    return _step(
        "D1.5",
        "measured",
        saved.get("achieved") is True,
        {"learned_prior": False, "n_parts": len(saved.get("parts") or [])},
        "Reduced the checkpoint p_i. learned_prior stays false.",
    )


def _run_d3(chain: Path, d2_status: str, mask_path: Path) -> dict[str, Any]:
    physics_path = chain / "physics.json"
    usda = chain / "physics_material.usda"
    gauss = chain / "gaussian_physics.npy"
    wrote = d2_status in {"filled", "reused"} and physics_path.is_file()
    if not wrote:
        return _step(
            "D3",
            "unstarted",
            False,
            {
                "physics_json": False,
                "physics_material_usda": False,
                "gaussian_physics_npy": False,
                "drop_test": "not_run",
                "drop_test_reason": DROP_NOT_SCHEDULED,
                "settle_within_2s": None,
                "learned_prior": False,
            },
            "No vision physics.json, so fuse, USD, and gaussian_physics.npy were not created. "
            + DROP_NOT_SCHEDULED
            + ".",
        )
    physics = _load(physics_path)
    if physics.get("learned_prior") is True:
        raise GateError("learned_prior cannot be true")
    settle = physics.get("settle_within_2s")
    _dbg200(
        "D",
        "tools/orbit_chain.py:_run_d3",
        "d3 settle",
        {
            "chain": chain.name,
            "drop_test": physics.get("drop_test") or "not_run",
            "drop_reason": physics.get("drop_test_reason"),
            "settle": settle,
            "isaac_called": False,
            "scheduled_reason": DROP_NOT_SCHEDULED,
        },
    )
    if settle is True:
        raise GateError("settle_within_2s cannot be true without telemetry")
    _schedule_drop(physics_path)
    physics = _load(physics_path)
    fused = fuse_physics(None, physics, None)
    if fused.get("learned_prior") is True:
        raise GateError("learned_prior cannot be true")
    write_physics_material_usda(usda, fused, prim="Can")
    prior_path = chain / "semantics" / "prior_input.npy"
    prior_rows = int(np.load(prior_path, mmap_mode="r").shape[0]) if prior_path.is_file() else 0
    n_rows = prior_rows or (int(np.load(mask_path).size) if mask_path.is_file() else 0)
    write_gaussian_physics(gauss, n_rows, None, physics)
    return _step(
        "D3",
        "partial",
        False,
        {
            "physics_json": True,
            "physics_material_usda": usda.is_file(),
            "gaussian_physics_npy": gauss.is_file(),
            "drop_test": "not_run",
            "drop_test_reason": DROP_NOT_SCHEDULED,
            "settle_within_2s": None,
            "learned_prior": False,
        },
        "Fused the VLM record. " + DROP_NOT_SCHEDULED + ".",
    )


def _d11_part(codebook: dict[str, Any], labeled: int | None = None) -> dict[str, Any]:
    vector = codebook.get("vector")
    dim = len(vector) if isinstance(vector, list) else None
    stale = codebook.get("n_copied")
    copied = labeled if isinstance(labeled, int) else stale
    reason = "Object vector already on disk. DINOv2 and OpenCLIP were not rerun. learned_prior stays false."
    if isinstance(labeled, int) and stale not in (None, labeled):
        reason = (
            "Object vector already on disk. n_copied is the object-label count on the index written this run. "
            "DINOv2 and OpenCLIP were not rerun. learned_prior stays false."
        )
    return _step(
        "D1.1",
        "measured" if dim else "unstarted",
        False,
        {
            "dim": dim,
            "n_copied": copied,
            "n_gaussians": codebook.get("n_gaussians"),
            "dinov2": codebook.get("dinov2") is True,
            "openclip": codebook.get("openclip") is True,
            "learned_prior": False,
        },
        reason,
    )


def _d13_part(chain: Path, status: dict[str, Any]) -> dict[str, Any]:
    path = chain / "semantics" / "prior_input.npy"
    numbers: dict[str, Any] = {
        "state": status.get("state"),
        "prior_input": path.is_file(),
        "learned_prior": False,
    }
    if path.is_file():
        array = np.load(path)
        numbers["rows"] = int(array.shape[0])
        numbers["cols"] = int(array.shape[1]) if array.ndim == 2 else None
    if status.get("ply_vertices") is not None:
        numbers["ply_vertices"] = status.get("ply_vertices")
    if status.get("matches_object_ply") is True:
        numbers["matches_object_ply"] = True
    achieved = False
    if path.is_file():
        sidecar = chain / "semantics" / "prior_input.json"
        if sidecar.is_file():
            achieved = _load(sidecar).get("achieved") is True
    return _step(
        "D1.3",
        "measured" if path.is_file() else "unstarted",
        achieved,
        numbers,
        _d13_reason(path, status),
    )


def _d13_reason(path: Path, status: dict[str, Any]) -> str:
    """A missing prior names the real gap. A present splat is not called missing."""
    if path.is_file():
        if status.get("matches_object_ply") is True:
            return "prior_input.npy matches the Block C object ply."
        return "prior_input.npy written from the splat."
    detail = str(status.get("reason") or "").strip()
    if detail in {"", "assigned vectors are missing"}:
        return (
            "No separate scene-mean vector, so assigned vectors are missing. "
            "prior_input.npy was not written."
        )
    return detail


def _d2_part(chain: Path, d2_status: str) -> dict[str, Any]:
    path = chain / "physics.json"
    physics = _load(path) if path.is_file() else {}
    achieved = physics.get("achieved") is True if path.is_file() else False
    if physics.get("mass_kg") is not None and d2_status not in {"filled", "reused"}:
        raise GateError("a guessed mass was written without a VLM fill")
    if d2_status == "reused":
        reason = "Existing VLM physics.json was reused. The model was not asked."
    elif d2_status in {"missing", "model not asked"}:
        reason = "No existing VLM physics.json. The model was not asked."
    else:
        reason = f"start_d2 returned {d2_status}."
    return _step(
        "D2",
        "measured" if path.is_file() and physics.get("mass_kg") is not None else "unstarted",
        achieved,
        {
            "d2_status": d2_status,
            "physics_json": path.is_file(),
            "material": physics.get("material"),
            "mass_kg": physics.get("mass_kg"),
            "learned_prior": False,
        },
        reason,
    )


def _e1_part(mesh: Path, status: dict[str, Any], splat: Path, mask: Path) -> dict[str, Any]:
    exists = mesh.is_file() and mesh.stat().st_size > 0
    score_path = mesh.parent.parent / "mesh_score.json"
    score = _load(score_path)
    chamfer = score.get("chamfer") if isinstance(score.get("chamfer"), dict) else {}
    achieved = bool(
        exists
        and status.get("achieved") is True
        and score.get("achieved") is True
        and chamfer.get("achieved") is True
    )
    return _step(
        "E1",
        "unstarted" if not exists else "partial",
        achieved,
        {
            "mesh": exists,
            "splat": splat.is_file(),
            "mask": mask.is_file(),
            "metres_per_unit": chamfer.get("metres_per_unit"),
            "chamfer_cm": chamfer.get("chamfer_cm"),
            "f1_at_1cm": chamfer.get("f1_at_1cm"),
            "splatfacto": False,
        },
        _e1_reason(status, splat, mask),
    )


def _e1_reason(status: dict[str, Any], splat: Path, mask: Path) -> str:
    """Name the file that is missing. A mask on disk is not described as missing."""
    reason = str(status.get("reason") or "")
    vague = "splat.ply or gaussian_in_mask.npy" in reason or reason in {"", "mesh.ply was not written."}
    if not vague:
        return reason
    if mask.is_file() and not splat.is_file():
        return "gaussian_in_mask.npy is on disk. splat.ply is missing, so mesh.ply was not written."
    if splat.is_file() and not mask.is_file():
        return "splat.ply is on disk. gaussian_in_mask.npy is missing, so mesh.ply was not written."
    if not splat.is_file() and not mask.is_file():
        return "splat.ply and gaussian_in_mask.npy are missing, so mesh.ply was not written."
    return reason or "mesh.ply was not written."


def _e2_part(folder: Path, parts: list[dict[str, Any]], hull: dict[str, Any]) -> dict[str, Any]:
    saved = _load(folder / "ingest" / "hulls.json")
    mesh = folder / "mesh" / "mesh.ply"
    mesh_exists = mesh.is_file() and mesh.stat().st_size > 0
    one_body = len(parts) == 1 and parts[0].get("id") == "body"
    built = saved.get("convex_hull") == "built" or hull.get("convex_hull") == "built"
    if not mesh_exists or not built:
        state = "unstarted"
        if not mesh_exists:
            reason = "No mesh.ply, so parts and hulls were not built. A one-body label without a mesh is not a decomposition."
        else:
            reason = "Hulls were not built. Missing part masks do not make E2 partial."
    elif built:
        state = "partial"
        reason = "Hulls were built from the mesh that E wrote."
    else:
        state = "unstarted"
        reason = "No part split was written."
    return _step(
        "E2",
        state,
        saved.get("achieved") is True and mesh_exists and built,
        {
            "one_body": one_body if mesh_exists else False,
            "n_parts": len(parts) if mesh_exists else 0,
            "convex_hull": saved.get("convex_hull") or hull.get("convex_hull"),
            "coacd": saved.get("coacd") is True,
            "n_hulls": saved.get("n_hulls"),
            "mesh": mesh_exists,
        },
        reason,
    )


def _e3_part(folder: Path, sweep: dict[str, Any]) -> dict[str, Any]:
    saved = _load(folder / "ingest" / "joint_sweep.json")
    source = saved or sweep
    ran = source.get("joint_sweep") == "ran" and _sweep_has_numbers(source)
    return _step(
        "E3",
        "not_run" if not ran else "measured",
        source.get("achieved") is True and source.get("m5_met") is True,
        {
            "joint_sweep": source.get("joint_sweep"),
            "n_joints": 0,
            "interpenetration": source.get("interpenetration"),
            "m5_met": source.get("m5_met"),
            "n_joints_swept": source.get("n_joints_swept"),
        },
        "A can has no moving part. The joint sweep is not_run.",
    )


def _e4_part(folder: Path, route: str | None) -> dict[str, Any]:
    path = folder / "deformable.json"
    saved = _load(path) if path.is_file() else {}
    physics = _load(folder / "physics.json")
    has_type = isinstance(physics.get("deformable_type"), str) and bool(physics.get("deformable_type"))
    if (saved.get("cloth") is True or saved.get("rope") is True or saved.get("soft") is True) and not has_type:
        raise GateError("a missing physics record must not invent cloth, rope, or soft")
    if not has_type or not path.is_file():
        return _step(
            "E4",
            "unstarted",
            False,
            {
                "route": None,
                "skipped": None,
                "cloth": False,
                "rope": False,
                "soft": False,
                "destruction": "not_run",
            },
            "No physics.json deformable_type, so the route was not set to rigid. No cloth, rope, or soft body was built.",
        )
    return _step(
        "E4",
        "partial" if saved.get("skipped") is True else "measured",
        saved.get("achieved") is True,
        {
            "route": saved.get("route") or route,
            "skipped": saved.get("skipped") is True,
            "cloth": saved.get("cloth") is True,
            "rope": saved.get("rope") is True,
            "soft": saved.get("soft") is True,
            "destruction": saved.get("destruction") or "not_run",
        },
        "Route recorded from D's deformable_type. No cloth, rope, or soft body was built.",
    )


def _orbit_intrinsics(can: Path, orbit: str) -> dict[str, Any] | None:
    """Read this orbit's camera. A missing transforms file is not a guessed focal length."""
    candidates = (
        can / "colmap" / orbit / "transforms.json",
        can / "colmap" / orbit / "intrinsics.json",
        can / "ingest" / f"intrinsics_{orbit}.json",
    )
    for path in candidates:
        record = _load(path)
        parsed = camera_intrinsics(record)
        if parsed is not None:
            return parsed
        frames = record.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    parsed = camera_intrinsics(frame)
                    if parsed is not None:
                        return parsed
    return None


def _f1_gaps(
    mesh: Path | None,
    capture: Path | None,
    mask: Path | None,
    world: Path | None,
    intrinsics: dict[str, Any] | None,
) -> list[str]:
    gaps: list[str] = []
    if mesh is None:
        gaps.append("mesh")
    if capture is None:
        gaps.append("capture")
    if mask is None:
        gaps.append("mask")
    if world is None:
        gaps.append("world_T_object")
    if intrinsics is None:
        gaps.append("intrinsics")
    return gaps


def _f1_reason(gaps: list[str], align: dict[str, Any]) -> str:
    sentences = {
        "mesh": "mesh.ply was not written by Block E.",
        "capture": "The review capture is missing.",
        "mask": "The review mask is missing.",
        "world_T_object": "world_T_object is missing because Block C was skipped.",
        "intrinsics": "Camera intrinsics for this orbit are missing.",
    }
    if gaps:
        return " ".join(sentences[name] for name in gaps)
    detail = align.get("achieved_reason") or align.get("reason") or ""
    return str(detail) if detail else "Alignment ran. achieved stays false until IoU and metres_per_unit both pass."


def _f1_part(folder: Path, align: dict[str, Any], gaps: list[str]) -> dict[str, Any]:
    path = folder / "align.json"
    saved = _load(path) if path.is_file() else {}
    has_world = "world_T_object" not in gaps
    achieved = bool(not gaps and path.is_file() and saved.get("achieved") is True and align.get("achieved") is True)
    if gaps:
        achieved = False
    state = "unstarted" if gaps or not path.is_file() else (saved.get("state") or align.get("state") or "measured")
    return _step(
        "F1",
        state,
        achieved,
        {
            "world_T_object": "present" if has_world else "missing because Block C was skipped",
            "missing": ",".join(gaps) if gaps else "",
            "converged": saved.get("converged") is True if path.is_file() else False,
            "iou": saved.get("iou") if path.is_file() else align.get("iou"),
            "metres_per_unit": saved.get("metres_per_unit") if path.is_file() else None,
            "align_json": path.is_file(),
        },
        _f1_reason(gaps, saved or align),
    )


def _f2_part(folder: Path, orient: dict[str, Any] | None, has_background: bool) -> dict[str, Any]:
    path = folder / "orientation.json"
    saved = _load(path) if path.is_file() else {}
    if not has_background:
        return _step(
            "F2",
            "unstarted",
            False,
            {"background_ply": False, "table_height_error_mm": None},
            "C did not write background.ply, so background orientation was not started.",
        )
    achieved = path.is_file() and saved.get("achieved") is True and orient is not None and orient.get("achieved") is True
    return _step(
        "F2",
        saved.get("state") or "measured",
        achieved,
        {
            "background_ply": True,
            "table_height_error_mm": saved.get("table_height_error_mm"),
            "skipped": saved.get("skipped") is True,
        },
        saved.get("reason") or "",
    )


def _f3_part(folder: Path, mesh_written: bool, has_asset: bool, has_physics: bool) -> dict[str, Any]:
    usdz = folder / "scene.usdz"
    text = (folder / "scene.usda").read_text(encoding="utf-8") if (folder / "scene.usda").is_file() else ""
    robot = "placed" if 'sceneagent:robot = "placed"' in text else "not_placed"
    if not mesh_written or not usdz.is_file():
        missing = [
            name
            for name, present in (("mesh", mesh_written), ("asset", has_asset), ("physics", has_physics))
            if not present
        ]
        detail = f" Missing {', '.join(missing)}." if missing else ""
        return _step(
            "F3",
            "unstarted",
            False,
            {
                "mesh": mesh_written,
                "asset": has_asset,
                "physics": has_physics,
                "scene_usdz": False,
                "robot": "not_placed",
                "franka": False,
            },
            f"scene.usdz was not written.{detail} No Franka asset was invented.",
        )
    return _step(
        "F3",
        "measured",
        False,
        {"scene_usdz": True, "robot": robot, "franka": "Franka" in text},
        "USDZ was written from E's mesh. Isaac has not scored it.",
    )


def _f4_part(folder: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = folder / "inspection_report.json"
    saved = _load(path) if path.is_file() else report
    settle = saved.get("settle_within_2s", None)
    if settle is True:
        raise GateError("settle_within_2s cannot be true without Isaac telemetry")
    state = saved.get("state") or "not_run"
    achieved = state == "measured" and saved.get("achieved") is True and saved.get("settle_ok") is True
    if state == "not_run":
        achieved = False
    return _step(
        "F4",
        state,
        achieved,
        {
            "settle_within_2s": settle,
            "settle_ok": saved.get("settle_ok"),
            "physx_warnings": saved.get("physx_warnings"),
            "moved_cm": saved.get("moved_cm"),
            "joint_limit_violations": saved.get("joint_limit_violations"),
        },
        saved.get("reason") or "Isaac was not run.",
    )


def _m6_part(folder: Path) -> dict[str, Any]:
    align = _load(folder / "align.json")
    orient = _load(folder / "orientation.json")
    iou = align.get("iou") if align.get("state") == "measured" else None
    error = orient.get("table_height_error_mm") if orient.get("skipped") is not True else None
    achieved = align.get("achieved") is True and orient.get("achieved") is True and iou is not None and error is not None
    if not (folder / "align.json").is_file() or not (folder / "orientation.json").is_file():
        achieved = False
    return _step(
        "M6",
        "unmeasured" if not achieved and (iou is None or error is None) else "measured",
        achieved,
        {"iou": iou, "table_height_error_mm": error, "metres_per_unit": align.get("metres_per_unit")},
        "M6 needs a Block F silhouette IoU and a table height error. mesh_score.json is not this IoU.",
    )


def _m7_part(folder: Path, report: dict[str, Any]) -> dict[str, Any]:
    frame = folder / "inspect" / "frame_000.png"
    numbers_ok = report.get("state") == "measured" and frame.is_file()
    achieved = numbers_ok and report.get("achieved") is True and report.get("settle_ok") is True
    return _step(
        "M7",
        "measured" if numbers_ok else "unmeasured",
        achieved,
        {
            "settle_ok": report.get("settle_ok") if numbers_ok else None,
            "physx_warnings": report.get("physx_warnings") if numbers_ok else None,
            "moved_cm": report.get("moved_cm") if numbers_ok else None,
            "settle_within_2s": None,
        },
        "M7 needs an Isaac frame and telemetry. Those numbers were not set.",
    )


def _m0_part(folder: Path) -> dict[str, Any]:
    saved = _load(folder / "ingest" / "scene_eval.json")
    f1 = saved.get("f1_at_1cm") if (folder / "ingest" / "scene_eval.json").is_file() else None
    chamfer = saved.get("chamfer_cm") if (folder / "ingest" / "scene_eval.json").is_file() else None
    achieved = saved.get("achieved") is True and f1 is not None and chamfer is not None
    return _step(
        "M0",
        "unmeasured" if f1 is None or chamfer is None else "measured",
        achieved,
        {"f1_at_1cm": f1, "chamfer_cm": chamfer, "metres_per_unit": saved.get("metres_per_unit")},
        "M0 needs a scene-level F1 and Chamfer. The can silhouette score is not F1.",
    )


def _b_part(step_id: str, score: dict[str, Any], check_step: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    part = _score_part(score, step_id)
    numbers = _scalars(check_step.get("numbers") if isinstance(check_step.get("numbers"), dict) else {})
    for key, value in _scalars(probe).items():
        numbers.setdefault(key, value)
    if check_step.get("mismatch"):
        numbers["mismatch"] = "; ".join(str(item) for item in check_step["mismatch"])
    if check_step.get("skipped"):
        numbers["skipped"] = "; ".join(str(item) for item in check_step["skipped"])
    numbers["learned_prior"] = False
    return _step(step_id, _b_state(part, check_step), _b_achieved(part, check_step), numbers, str(part.get("reason") or ""))


def _milestone_m1(score: dict[str, Any], check: dict[str, Any], step_a: dict[str, Any]) -> dict[str, Any]:
    numbers = _scalars(step_a.get("numbers") if isinstance(step_a.get("numbers"), dict) else {})
    m1 = score.get("m1") if isinstance(score.get("m1"), dict) else {}
    numbers["m1_met"] = score.get("m1_met") is True
    numbers["m1_achieved"] = score.get("m1_achieved") is True
    if "metres_per_unit" not in numbers:
        numbers["metres_per_unit"] = m1.get("metres_per_unit")
    blocked = bool(step_a.get("skipped") or step_a.get("mismatch"))
    file_says = (score.get("m1_achieved") is True or score.get("m1_met") is True) and check.get("m1_met") is True
    state = "measured" if numbers.get("psnr") is not None else "unstarted"
    if step_a.get("skipped"):
        state = "skipped"
    return _step(
        "M1",
        state,
        file_says and not blocked,
        numbers,
        str(m1.get("reason") or "M1 stays unmet while metres_per_unit is missing."),
    )


def _milestone_m2(score: dict[str, Any], check: dict[str, Any], step_b4: dict[str, Any]) -> dict[str, Any]:
    numbers = _scalars(step_b4.get("numbers") if isinstance(step_b4.get("numbers"), dict) else {})
    numbers["m2_met"] = score.get("m2_met") is True
    numbers["m2_achieved"] = score.get("m2_achieved") is True
    blocked = bool(step_b4.get("skipped") or step_b4.get("mismatch"))
    file_says = (score.get("m2_achieved") is True or score.get("m2_met") is True) and check.get("m2_met") is True
    state = "unmeasured" if numbers.get("miou") is None else "measured"
    return _step(
        "M2",
        state,
        file_says and not blocked,
        numbers,
        str(score.get("m2_reason") or "M2 needs mIoU on 20 hand-labelled views."),
    )


def _c_record(folder: Path, orbit: str) -> dict[str, Any]:
    """Read ingest/block_c.json. A missing file or orbit stays skipped."""
    path = Path(folder) / "ingest" / "block_c.json"
    if not path.is_file():
        return _skip_c(orbit)
    loaded = _load(path)
    orbits = loaded.get("orbits") if isinstance(loaded.get("orbits"), dict) else {}
    row = orbits.get(orbit)
    if not isinstance(row, dict):
        return _skip_c(orbit)
    return _c_from_row(orbit, row)


def _c_from_row(orbit: str, row: dict[str, Any]) -> dict[str, Any]:
    """Copy the stored C row. achieved is true only when that file's M3 is true."""
    from tools.block_c import m3_met

    met = m3_met(row) and row.get("m3_met") is True
    state = str(row.get("state") or "")
    crop = row.get("crop")
    if crop == "refused":
        step_state = "measured"
        block_state = "measured"
        met = False
    elif crop == "object_only":
        step_state = "measured"
        block_state = "measured"
        met = False
    elif state == "skipped" or row.get("infill") == "skipped":
        step_state = "skipped"
        block_state = "skipped"
        met = False
    elif met:
        step_state = "measured"
        block_state = "measured"
    else:
        step_state = "partial"
        block_state = "partial"
    holes = row.get("holes") if isinstance(row.get("holes"), dict) else {}
    vlm = row.get("vlm") if isinstance(row.get("vlm"), dict) else {}
    numbers = {
        "floater_count": row.get("floater_count"),
        "n_object": row.get("n_object"),
        "n_background": row.get("n_background"),
        "n_inside": row.get("n_inside"),
        "n_outside": row.get("n_outside"),
        "crop": row.get("crop"),
        "covers_room": row.get("covers_room"),
        "feature_field": row.get("feature_field"),
        "n_hole_masks": holes.get("n_masks"),
        "infill": row.get("infill"),
        "fine_tune": row.get("fine_tune"),
        "psnr_drop_db": row.get("psnr_drop_db"),
        "m3_met": met,
        "vlm_asked": vlm.get("asked"),
        "vlm_verdict": vlm.get("verdict"),
        "seams": vlm.get("seams"),
    }
    if crop == "refused":
        reason = str(row.get("note") or "The mask covers the room. The crop was not written. It is not a pass.")
    elif crop == "object_only":
        reason = str(row.get("note") or "Infill skipped. The object is the scene. It is not a pass.")
        if "not a pass" not in reason:
            reason = f"{reason} It is not a pass."
    elif step_state == "skipped":
        reason = str(row.get("note") or "Block C was not run. It is not a pass. world_T_object was not written.")
    elif met:
        reason = str(row.get("note") or "M3 met.")
    else:
        reason = str(row.get("note") or "Infill did not pass. M3 is not met.")
    parts = [_step("C", step_state, met, numbers, reason)]
    return _block_record("C", orbit, parts, state=block_state)


def replace_orbit_c_section(markdown: str, orbit: str, section: str) -> str:
    """Replace the ### C block under one orbit heading. Other blocks stay."""
    lines = markdown.splitlines(keepends=True)
    heading = f"## {orbit}\n"
    start = None
    for index, line in enumerate(lines):
        if line in {heading, heading.replace("\n", "\r\n")}:
            start = index + 1
            break
    if start is None:
        return markdown
    c_at = None
    next_orbit = len(lines)
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("## "):
            next_orbit = index
            break
        if stripped.startswith("### C"):
            c_at = index
            break
    if c_at is None:
        return markdown
    end = next_orbit
    for index in range(c_at + 1, next_orbit):
        if lines[index].strip().startswith("### "):
            end = index
            break
    body = section if section.endswith("\n") else section + "\n"
    rebuilt = "".join(lines[:c_at]) + body + "".join(lines[end:])
    return rebuilt


def refresh_c_line(folder: Path, markdown: Path, orbit: str) -> bool:
    """Rewrite one C section from ingest/block_c.json. A missing markdown stays missing."""
    path = Path(markdown)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    section = render_block(_c_record(folder, orbit))
    updated = replace_orbit_c_section(text, orbit, section)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _skip_c(orbit: str) -> dict[str, Any]:
    parts = [
        _step(
            "C",
            "skipped",
            False,
            {},
            "Block C was not run. It is not a pass. world_T_object was not written.",
        )
    ]
    return _block_record("C", orbit, parts, state="skipped")


def _b_inputs(can: Path, orbit: str) -> dict[str, Path]:
    codebook = can / ("codebook.json" if orbit == "orbit1" else f"codebook_{orbit}.json")
    return {
        "object": can / "masks" / orbit / "object.json",
        "mask": can / "masks" / orbit / "gaussian_in_mask.npy",
        "codebook": codebook,
        "instances": can / "semantics" / orbit / "instances.json",
        "splat": can / "colmap" / orbit / "splat.ply",
    }


def _d2_frames(can: Path) -> list[Path]:
    frames = []
    for parts in _D2_FRAMES:
        frames.append(can.joinpath(*parts))
    return frames


def _bbox(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_inside": _finite(obj.get("span_inside")),
        "span_scene": _finite(obj.get("span_scene")),
        "units": "colmap",
        "metres_per_unit": None,
    }


def _checkpoint(can: Path, chain: Path) -> Path | None:
    for path in (
        chain / "semantics" / "physics_prior.npz",
        can / "semantics" / "physics_prior.npz",
        can / "weights" / "physics_prior.npz",
    ):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _b_achieved(score_part: dict[str, Any], check_step: dict[str, Any]) -> bool:
    if check_step.get("skipped") or check_step.get("mismatch"):
        return False
    if score_part.get("achieved") is not True:
        return False
    if "achieved_claimed" in check_step and check_step.get("achieved_claimed") is not True:
        return False
    return True


def _b_state(score_part: dict[str, Any], check_step: dict[str, Any]) -> str:
    if check_step.get("skipped"):
        return "skipped"
    state = score_part.get("state")
    if isinstance(state, str) and state:
        return state
    if check_step.get("ok") is True:
        return "measured"
    return "unstarted"


def _score_part(score: dict[str, Any], step_id: str) -> dict[str, Any]:
    for part in score.get("parts") or []:
        if isinstance(part, dict) and part.get("id") == step_id:
            return part
    return {}


def _step(step_id: str, state: str, achieved: bool, numbers: dict[str, Any], reason: str) -> dict[str, Any]:
    clean = {key: _scalar(value) for key, value in numbers.items()}
    return {
        "id": step_id,
        "state": state,
        "achieved": achieved is True,
        "numbers": clean,
        "reason": reason,
    }


def _unmeasured(step_id: str, reason: str) -> dict[str, Any]:
    return _step(step_id, "unmeasured", False, {"achieved": False}, reason)


def _block_record(block: str, orbit: str, parts: list[dict[str, Any]], state: str | None = None) -> dict[str, Any]:
    achieved = bool(parts) and all(part.get("achieved") is True for part in parts)
    return {
        "block": block,
        "object": "can",
        "orbit": orbit,
        "state": state or _roll_state(parts),
        "achieved": achieved,
        "learned_prior": False,
        "parts": parts,
    }


def _e_block_state(folder: Path) -> str:
    """Partial only when mesh.ply exists. A default one-body route is not a mesh."""
    mesh = folder / "mesh" / "mesh.ply"
    if mesh.is_file() and mesh.stat().st_size > 0:
        return "partial"
    return "not_started"


def _f_block_state(folder: Path, parts: list[dict[str, Any]]) -> str:
    """No world_T_object, from skipping C, keeps Block F not started."""
    world = _first_file(
        folder,
        ("world_T_object.json", "ingest/world_T_object.json", "objects/can/world_T_object.json"),
    )
    if world is None:
        return "not_started"
    return _roll_state(parts)


def _roll_state(parts: list[dict[str, Any]]) -> str:
    states = [str(part.get("state")) for part in parts]
    active = {"measured", "partial", "written", "ran"}
    idle = {"unstarted", "skipped", "not_run", "unmeasured", "not_started", "not_scheduled"}
    if any(item in active for item in states) and any(item in idle for item in states):
        return "partial"
    if states and all(item in idle for item in states):
        return "not_started"
    if states and all(item in active for item in states):
        return "partial"
    return "partial"


def _append_rendered(markdown: Path, record_path: Path) -> None:
    text = render_block_file(record_path)
    with Path(markdown).open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _write_record(path: Path, record: dict[str, Any]) -> Path:
    if record.get("learned_prior") is True:
        raise GateError("learned_prior cannot be true")
    record = dict(record)
    record["learned_prior"] = False
    _dump(path, record)
    return path


def _fresh_chain(can: Path, orbit: str) -> Path:
    chain = (can / "chain" / orbit).resolve()
    root = (can / "chain").resolve()
    if chain != root and root not in chain.parents:
        raise RuntimeError(f"{chain} is outside {root}")
    if chain.is_symlink():
        raise RuntimeError("refusing to reset a symlinked chain folder")
    if chain.exists():
        shutil.rmtree(chain)
    chain.mkdir(parents=True)
    return chain


def _require_can(can: Path) -> Path:
    can = Path(can)
    if can.name != "can":
        raise ValueError("orbit chain only runs on the can object folder")
    if any(part in {"bottle", "vaseline"} for part in can.parts):
        raise ValueError("bottle and vaseline stay untouched")
    return can


def _require_chain(can: Path, folder: Path) -> Path:
    folder = Path(folder).resolve()
    root = (Path(can) / "chain").resolve()
    if folder != root and root not in folder.parents:
        raise RuntimeError(f"{folder} is outside {root}")
    if folder == Path(can).resolve():
        raise RuntimeError("refusing to write the shared can folder")
    return folder


def _require_inside(path: Path, parent: Path) -> None:
    resolved = Path(path).resolve()
    parent_resolved = Path(parent).resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise RuntimeError(f"{resolved} is outside {parent_resolved}")


def _copy_produced(produced: Path, chain: Path) -> None:
    if not produced.is_dir():
        return
    for path in produced.rglob("*"):
        if not path.is_file():
            continue
        dest = chain / path.relative_to(produced)
        _require_inside(dest, chain)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _snapshot(can: Path) -> dict[str, bytes | None]:
    snap: dict[str, bytes | None] = {}
    for rel in _PROTECTED:
        path = can / rel
        snap[rel] = path.read_bytes() if path.is_file() else None
    return snap


def _guard_shared(can: Path, snap: dict[str, bytes | None]) -> None:
    changed: list[str] = []
    for rel, previous in snap.items():
        path = can / rel
        current = path.read_bytes() if path.is_file() else None
        if current == previous:
            continue
        changed.append(rel)
        if previous is None:
            if path.is_file():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(previous)
    if changed:
        raise RuntimeError("refusing to keep writes on shared can files: " + ", ".join(changed))


def _brief_orbit(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "B": {"state": item["b"].get("state"), "achieved": item["b"].get("achieved") is True},
        "C": {"state": "skipped", "achieved": False},
        "D": {"state": item["d"].get("state"), "achieved": item["d"].get("achieved") is True},
        "E": {"state": item["e"].get("state"), "achieved": item["e"].get("achieved") is True},
        "F": {"state": item["f"].get("state"), "achieved": item["f"].get("achieved") is True},
    }


def _achieved_cell(value: Any) -> str:
    if value is True:
        return "achieved: true"
    return "achieved: false"


def _numbers_text(part: dict[str, Any]) -> str:
    bits = [f"{key}={_fmt(value)}" for key, value in (part.get("numbers") or {}).items()]
    reason = part.get("reason") or ""
    if reason:
        bits.append(f"note={reason}")
    return "; ".join(bits)


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "null"
        return f"{value:.6g}"
    return str(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _scalars(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (list, dict, tuple)):
            continue
        out[str(key)] = _scalar(value)
    return out


def copy_c1_pose(can: Path, orbit: str, chain: Path) -> Path | None:
    """Copy objects/can/<orbit>/world_T_object.json into the chain folder.

    Returns that chain path when C1 wrote a pose. A missing file stays missing.
    This does not write align.json and does not invent IoU.
    """
    orbit = _orbit(orbit)
    dest = Path(chain) / "world_T_object.json"
    if dest.is_file():
        return dest
    source = Path(can) / "objects" / "can" / orbit / "world_T_object.json"
    if not source.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())
    return dest


def _orbit_background(can: Path, orbit: str, folder: Path) -> Path | None:
    found = _first_file(folder, ("background.ply", "background/splat.ply"))
    if found is not None:
        return found
    orbit_background = can / "background" / orbit / "background.ply"
    if orbit_background.is_file():
        return orbit_background
    shared = can / "background.ply"
    return shared if shared.is_file() else None


def _first_file(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.is_file():
            return path
    return None


def _sweep_has_numbers(record: dict[str, Any]) -> bool:
    swept = record.get("n_joints_swept")
    if isinstance(swept, bool) or not isinstance(swept, int):
        return False
    return isinstance(record.get("interpenetration"), bool) and _finite(record.get("max_penetration_m")) is not None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _orbit(orbit: str) -> str:
    if orbit not in ORBITS:
        raise ValueError(f"can orbit chain only runs {ORBITS}")
    return orbit


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dump(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
