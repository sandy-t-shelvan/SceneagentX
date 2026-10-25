"""Run SceneAgent build blocks in order.

B0 and B1 are the laptop checks. B2 to B4 call COLMAP, splatfacto, and
alignment. Later blocks stop with a clear message instead of pretending
the rest of the GPU tools ran.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

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
from tools.block_b_job import caption_from_spec, shell_quote
from tools.common.paths import object_dir, repo_root
from tools.task_spec import accept_pipeline_prompt, failing_room_orbits, gate_detected_target, record_room_mask_retry
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
    prompt: str | None = None,
) -> int:
    if prompt is not None:
        code = accept_pipeline_prompt(obj, prompt, root, source=source)
        if code != 0:
            return code
        code = gate_detected_target(obj, root)
        if code != 0:
            return code
    code = apply_mask_retry(obj, root)
    if code != 0:
        return code
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


def mask_retry_commands(obj: str, root: Path | None = None) -> list[list[str]]:
    """Detection and lift for the orbits named in mask_retry.json.

    object_semantics.py receives --box-threshold. The splat is not in the list,
    and the feature field is not started.
    """
    folder = object_dir(obj, root)
    retry = _read_retry(folder)
    if retry is None or retry.get("detection") == "rerun":
        return []
    try:
        spec = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(spec, dict):
        return []
    params = retry.get("params") if isinstance(retry.get("params"), dict) else {}
    prompt_override = params.get("text_prompt")
    if isinstance(prompt_override, str) and prompt_override.strip():
        caption = prompt_override.strip()
    else:
        try:
            caption = caption_from_spec(spec)
        except ValueError:
            return []
    target = spec.get("target")
    target_text = target.strip() if isinstance(target, str) else ""
    orbits = _retry_orbits(folder, retry)
    if not orbits:
        return []
    root_text = str(folder.parent.parent)
    semantics = str(repo_root() / "vast-cluster" / "scripts" / "object_semantics.py")
    lift = str(repo_root() / "vast-cluster" / "scripts" / "object_crop_iou.py")
    commands: list[list[str]] = []
    for orbit in orbits:
        detect = [semantics, "--obj", obj, "--prompt", caption, "--orbit", orbit, "--root", root_text]
        if target_text:
            detect.extend(["--target", target_text])
        threshold = params.get("box_threshold")
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            detect.extend(["--box-threshold", f"{float(threshold):.2f}"])
        config = _splat_config(folder, orbit)
        if not config:
            continue
        commands.append(detect)
        commands.append(
            [lift, "--obj", obj, "--prompt", caption, "--orbit", orbit, "--config", config, "--root", root_text]
        )
    return commands


def apply_mask_retry(
    obj: str,
    root: Path | None = None,
    runner: Callable[[list[str]], int] | None = None,
    host_lookup: Callable[[], Any] | None = None,
) -> int:
    """Rerun detection and lift once when mask_retry.json is still pending.

    No running Vast host leaves the retry on disk and does not claim a new mask.
    A second room-sized lift stays refused. The splat is not retrained.
    """
    folder = object_dir(obj, root)
    path = folder / "mask_retry.json"
    if not path.is_file():
        return 0
    retry = _read_retry(folder)
    if retry is None:
        sys.stderr.write("mask_retry.json could not be read\n")
        return 2
    if retry.get("retrain") is True or retry.get("splat_retrained") is True:
        sys.stderr.write("mask retry refuses to retrain the splat\n")
        return 2
    if retry.get("detection") == "rerun":
        return _stop_if_still_room(folder)
    commands = mask_retry_commands(obj, root)
    if not commands:
        if failing_room_orbits(folder):
            sys.stderr.write("mask retry has no lift config; the mask was not claimed\n")
        return 0
    pull = runner is None
    host = None
    if runner is None:
        lookup = host_lookup or running_vast_host
        try:
            host = lookup()
        except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"{exc}\n")
            return 0
        if host is None:
            return 0
        runner = lambda argv, host=host: ssh_retry(host, argv)
    for argv in commands:
        code = int(runner(list(argv)))
        if code != 0:
            return code
        if pull and host is not None and Path(argv[0]).name == "object_crop_iou.py":
            orbit = argv[argv.index("--orbit") + 1]
            copied = pull_lift(host, obj, orbit, root)
            if copied != 0:
                sys.stderr.write("the new mask was not copied back; it is not claimed\n")
                return copied
    retry["detection"] = "rerun"
    retry["retrain"] = False
    retry["splat_retrained"] = False
    path.write_text(json.dumps(retry, indent=2) + "\n", encoding="utf-8")
    return _stop_if_still_room(folder)


def running_vast_host():
    """The one running Vast instance. A stopped host is not reused."""
    from tools.vast_host import VastError, VastHost, list_instances

    try:
        rows = list_instances()
    except (VastError, OSError, json.JSONDecodeError, ValueError):
        return None
    running = [row for row in rows if isinstance(row, dict) and str(row.get("actual_status")) == "running"]
    if len(running) != 1:
        return None
    chosen = running[0]
    host = str(chosen.get("public_ipaddr") or "")
    try:
        port = int(chosen.get("direct_port_start") or 0)
        instance_id = int(chosen["id"])
    except (KeyError, TypeError, ValueError):
        return None
    if not host or port <= 0:
        return None
    return VastHost(instance_id=instance_id, status="running", host=host, port=port)


def ssh_retry(host, argv: list[str]) -> int:
    """Run one detection or lift command on the current Vast host."""
    script = Path(argv[0]).name
    python = _REMOTE_PYTHON.get(script)
    if python is None:
        sys.stderr.write(f"mask retry does not run {script}\n")
        return 2
    remote = [python, "-u", f"/root/SceneagentX/vast-cluster/scripts/{script}"]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--root":
            remote.extend(["--root", "/root/SceneagentX"])
            index += 2
            continue
        remote.append(token)
        index += 1
    key = Path.home() / ".ssh" / "ssh_sceneagentX_1"
    if not key.is_file():
        sys.stderr.write("ssh key is missing\n")
        return 2
    quoted = " ".join(shell_quote(part) for part in remote)
    done = subprocess.run(
        [
            "ssh",
            "-i",
            str(key),
            "-p",
            str(host.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "BatchMode=yes",
            f"root@{host.host}",
            f"cd /root/SceneagentX && {quoted}",
        ],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        sys.stderr.buffer.write(done.stderr or b"")
    else:
        sys.stdout.buffer.write(done.stdout or b"")
    return int(done.returncode)


def pull_lift(host, obj: str, orbit: str, root: Path | None) -> int:
    """Copy the new lift back. A failed copy does not claim the mask."""
    try:
        name = _safe_token(obj)
        orbit_id = _safe_token(orbit)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    key = Path.home() / ".ssh" / "ssh_sceneagentX_1"
    local_dir = object_dir(name, root) / "masks" / orbit_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("object.json", "gaussian_in_mask.npy"):
        remote = f"/root/SceneagentX/data/{name}/masks/{orbit_id}/{filename}"
        local = local_dir / filename
        partial = local.with_name(local.name + ".partial")
        done = subprocess.run(
            [
                "scp",
                "-i",
                str(key),
                "-P",
                str(host.port),
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "BatchMode=yes",
                f"root@{host.host}:{remote}",
                str(partial),
            ],
            capture_output=True,
            check=False,
        )
        if done.returncode != 0:
            sys.stderr.buffer.write(done.stderr or b"")
            if partial.is_file():
                partial.unlink()
            return int(done.returncode)
        partial.replace(local)
    return 0


def _stop_if_still_room(folder: Path) -> int:
    recorded = record_room_mask_retry(folder)
    if recorded.get("action") != "stop":
        return 0
    bits = []
    for row in recorded.get("measured") or []:
        if isinstance(row, dict):
            bits.append(
                f"{row.get('orbit')} span_inside {row.get('span_inside')} span_scene {row.get('span_scene')}"
            )
    detail = "; ".join(bits)
    line = "target mask stayed room-sized"
    if detail:
        line += f" ({detail})"
    sys.stderr.write(line + "; the retry was not repeated\n")
    return 2


def _read_retry(folder: Path) -> dict[str, Any] | None:
    path = folder / "mask_retry.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _retry_orbits(folder: Path, retry: dict[str, Any]) -> list[str]:
    listed = retry.get("orbits")
    if not isinstance(listed, list):
        listed = [row["orbit"] for row in failing_room_orbits(folder)]
    orbits = []
    for item in listed:
        if not isinstance(item, str):
            continue
        try:
            orbits.append(_safe_token(item))
        except ValueError:
            continue
    return orbits


def _splat_config(folder: Path, orbit: str) -> str | None:
    path = folder / "ingest" / "block_a.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    for row in loaded.get("orbits") or []:
        if isinstance(row, dict) and row.get("orbit") == orbit and row.get("config"):
            return str(row["config"])
    return None


def _safe_token(text: str) -> str:
    if text != Path(text).name or text in {"", ".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"refusing name {text}")
    return text


_REMOTE_PYTHON = {
    "object_semantics.py": "/root/.local/share/mamba/envs/gsam2/bin/python",
    "object_crop_iou.py": "/root/.local/share/mamba/envs/nerf/bin/python",
}


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
    parser.add_argument("--prompt", default=None, help="The user's pipeline prompt, stored unchanged")
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
            prompt=args.prompt,
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
