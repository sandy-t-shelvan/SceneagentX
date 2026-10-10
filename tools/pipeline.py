"""Run SceneAgent build blocks in order.

B0 and B1 are the laptop checks. B2 to B4 call COLMAP, splatfacto, and
alignment. Later blocks stop with a clear message instead of pretending
the rest of the GPU tools ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from tools.agent_loop import SkillFault, choose_path, run_skill
from tools.align import run_align
from tools.check_env import Runner, run_b0
from tools.common.blocks import blocks_through, get_block
from tools.common.status import StatusError, load_status, save_status, set_path
from tools.ingest import COLMAP_BLOCKS, InputError, classify_input, colmap_refused, ingest_mesh, refusal_message
from tools.mesh_stages import GPU_ONLY, run_mesh_through
from tools.video_input import find_video, record_absent
from tools.step00_check_frames import check_object
from tools.step01_colmap import run_orbit as run_colmap_orbit
from tools.step02_splatfacto import run_orbit as run_splat_orbit


def run(
    obj: str,
    through: str = "B1",
    root: Path | None = None,
    runner: Runner | None = None,
    source: Path | None = None,
    length_m: float | None = None,
    n_gaussians: int = 500_000,
    n_views: int = 8,
) -> int:
    last = get_block(through)
    doc = load_status(obj, root, create=True)
    if source is not None:
        try:
            record = classify_input(source)
        except InputError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        set_path(doc, "input", record)
        save_status(obj, doc, root)
        if record["kind"] == "polycam_mesh":
            try:
                result = ingest_mesh(
                    obj,
                    Path(record["mesh"]),
                    root,
                    length_m=length_m,
                    n_gaussians=n_gaussians,
                    n_views=n_views,
                )
            except (OSError, ValueError, InputError) as exc:
                sys.stderr.write(f"{exc}\n")
                return 2
            sys.stdout.write(
                f"polycam_mesh  gaussians {result['n_gaussians']}  "
                f"synthetic views {result['n_synthetic_views']}  COLMAP not run\n"
            )
            if last.id in COLMAP_BLOCKS:
                sys.stderr.write(refusal_message())
                return 2
            if last.id in GPU_ONLY:
                code = run_mesh_through(obj, "B14", root)
                if code != 0:
                    return code
                sys.stderr.write(
                    f"{last.id} needs Isaac Lab or a robot and is not in this repo. "
                    "The mesh scene is packaged through B14.\n"
                )
                return 3
            if _at_or_after(last.id, "B4"):
                return run_mesh_through(obj, last.id, root)
            return 0
    for block in blocks_through(last.id):
        if block.id == "B0":
            code = run_b0(obj, root, runner)
        elif block.id == "B1":
            code = check_object(obj, root)
        elif block.id in COLMAP_BLOCKS and colmap_refused(load_status(obj, root)):
            sys.stderr.write(refusal_message())
            return 2
        elif block.id in COLMAP_BLOCKS and find_video(obj, root) is None:
            record_absent(obj, root)
            sys.stderr.write(
                "COLMAP and splatfacto require data/<obj>/raw/orbit1.mp4. "
                "They were not started on this input.\n"
            )
            return 2
        elif block.id == "B2":
            code, _called = run_skill(
                obj,
                "B2",
                lambda _parameter: _video_program(run_colmap_orbit, obj, root, runner),
                root,
            )
        elif block.id == "B3":
            code, _called = run_skill(
                obj,
                "B3",
                lambda _parameter: _video_program(run_splat_orbit, obj, root, runner),
                root,
            )
        elif block.id == "B4":
            code = run_align(obj, root, runner)
        else:
            sys.stderr.write(
                f"{block.id} ({block.name}) is not implemented. "
                f"Reconstruction stops at B4. This block needs the GPU tools listed in env/{_env_for(block.id)}.yml "
                "and third-party checkpoints that are not vendored in this repo.\n"
            )
            return 3
        if code != 0:
            sys.stdout.flush()
            sys.stderr.write(f"Stopped at {block.id}. Later blocks were not started.\n")
            return code
    return 0


def _video_program(program, obj: str, root: Path | None, runner: Runner | None) -> None:
    if choose_path(obj, root) == "ply":
        raise SkillFault("COLMAP is refused for a ply")
    code = program(obj, 1, root, runner)
    if code != 0:
        raise SkillFault(f"{program.__name__} exited {code}")


def _env_for(block_id: str) -> str:
    return {
        "B2": "nerf",
        "B3": "nerf",
        "B4": "nerf",
        "B5": "gsam2",
        "B6": "nerf",
        "B7": "sa",
        "B8": "2dgs",
        "B9": "partfield",
        "B10": "sa",
        "B11": "physgauss",
        "B12": "sa",
        "B13": "sa",
        "B14": "trellis",
        "B15": "physgauss",
        "B16": "sa",
        "B17": "sa",
    }.get(block_id, "sa")


def _at_or_after(block_id: str, gate_id: str) -> bool:
    from tools.common.blocks import BLOCKS

    order = [block.id for block in BLOCKS]
    return order.index(block_id) >= order.index(gate_id)


def _print_blocks() -> None:
    from tools.common.blocks import BLOCKS

    for block in BLOCKS:
        if block.id in GPU_ONLY:
            implemented = "needs robot"
        elif block.phase <= 2 or block.id in {
            "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14"
        }:
            implemented = "ready"
        else:
            implemented = "later"
        sys.stdout.write(f"{block.id}  phase {block.phase}  {implemented}  {block.summary}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SceneAgent reproduction blocks")
    parser.add_argument("--obj", default="marker")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--through", default="B1", help="Last block to run. B2 to B4 are reconstruction.")
    parser.add_argument("--input", type=Path, default=None, help="Video, image folder, or a Polycam .obj/.ply/.gltf mesh")
    parser.add_argument("--length-m", type=float, default=None, help="Ruler length of the mesh's longest side, in metres")
    parser.add_argument("--n-gaussians", type=int, default=500_000)
    parser.add_argument("--list", action="store_true", help="List blocks and exit")
    args = parser.parse_args(argv)
    if args.list:
        _print_blocks()
        return 0
    try:
        code = run(
            args.obj,
            args.through,
            args.root,
            source=args.input,
            length_m=args.length_m,
            n_gaussians=args.n_gaussians,
        )
    except (StatusError, KeyError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    try:
        _refresh_changelog(args.obj, args.root)
    except (StatusError, KeyError, RuntimeError, OSError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return code


def _refresh_changelog(obj: str, root: Path | None) -> None:
    from tools.changelog import write_changelog
    from tools.common.paths import status_path

    if not status_path(obj, root).is_file():
        return
    path = write_changelog(obj, root)
    sys.stdout.write(f"changelog  {path}\n")


if __name__ == "__main__":
    raise SystemExit(main())
