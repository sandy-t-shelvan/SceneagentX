"""Run SceneAgent build blocks in order.

B1 stays on the laptop. B0, COLMAP, splatfacto, and alignment run on the
one running Vast machine. Later blocks stop with a clear message instead
of pretending the rest of the GPU tools ran.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from tools.agent_loop import SkillFault, choose_path, run_skill
from tools.align import run_align
from tools.check_env import CmdResult, Runner, run_b0
from tools.common.blocks import blocks_through, get_block
from tools.common.status import StatusError, begin_step, finish_step, load_status, save_status, set_path
from tools.ingest import COLMAP_BLOCKS, InputError, classify_input, colmap_refused, ingest_mesh, refusal_message
from tools.mesh_stages import GPU_ONLY, run_mesh_through
from tools.block_b_job import caption_from_spec, detector_caption, shell_quote
from tools.common.paths import object_dir, raw_dir, repo_root, resolve_root
from tools.task_spec import accept_pipeline_prompt, failing_room_orbits, gate_detected_target, record_room_mask_retry
from tools.vast_host import VastHost
from tools.video_input import VIDEO_SUFFIXES, find_video, record_absent
from tools.step00_check_frames import check_object
from tools.step01_colmap import run_orbit as run_colmap_orbit
from tools.step02_splatfacto import passing_checkpoint_stays, run_orbit as run_splat_orbit


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
    held = False

    def hold(gate: str, detail: str, check: str = "unmet") -> bool:
        nonlocal held
        if _bypass_stop(obj, root, gate, detail, check=check):
            held = True
            return True
        return False

    if prompt is not None:
        code = accept_pipeline_prompt(obj, prompt, root, source=source)
        if code != 0 and not hold("pipeline_prompt", "the pipeline prompt was not stored"):
            return code
        code = gate_detected_target(obj, root)
        if code != 0 and not hold("room_mask", "target mask check stopped"):
            return code
    code = apply_mask_retry(obj, root)
    if code != 0 and not hold("room_mask_retry", "the room-mask retry stopped"):
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
    vast: VastRunner | None = None
    active: Runner | None = runner
    if runner is None:
        host = running_vast_host()
        if host is not None:
            vast = VastRunner(host, resolve_root(root))
            active = vast
    for block in blocks_through(last.id):
        if block.id == "B0":
            if runner is None and vast is None:
                code = _fail_b0_no_vast(obj, root)
            else:
                code = run_b0(obj, root, active)
        elif block.id == "B1":
            code = check_object(obj, root)
        elif block.id in COLMAP_BLOCKS and colmap_refused(load_status(obj, root)):
            sys.stderr.write(refusal_message())
            code = 2
        elif block.id in COLMAP_BLOCKS and find_video(obj, root) is None:
            record_absent(obj, root)
            sys.stderr.write(
                "COLMAP and splatfacto require data/<obj>/raw/orbit1.mp4. "
                "They were not started on this input.\n"
            )
            code = 2
        elif block.id == "B2":
            code = push_missing_raw_videos(vast.host, obj, root) if vast is not None else 0
            if code == 0:
                code, _called = run_skill(
                    obj,
                    "B2",
                    lambda _parameter: _video_program(run_colmap_orbit, obj, root, active),
                    root,
                )
        elif block.id == "B3":
            kept = passing_checkpoint_stays(obj, 1, root)
            code, _called = run_skill(
                obj,
                "B3",
                lambda _parameter: _video_program(run_splat_orbit, obj, root, active),
                root,
            )
            if code == 0 and vast is not None and not kept and not vast.copied_splat:
                code = pull_reconstruction(vast.host, obj, root, 1)
        elif block.id == "B4":
            code = run_align(obj, root, active)
        else:
            sys.stderr.write(
                f"{block.id} ({block.name}) is not implemented. "
                f"Reconstruction stops at B4. This block needs the GPU tools listed in env/{_env_for(block.id)}.yml "
                "and third-party checkpoints that are not vendored in this repo.\n"
            )
            code = 3
        if code != 0:
            gate, detail, check = _block_gate(block.id, code)
            if hold(gate, detail, check):
                continue
            sys.stdout.flush()
            sys.stderr.write(f"Stopped at {block.id}. Later blocks were not started.\n")
            return code
    return 1 if held else 0


def _bypass_stop(obj: str, root: Path | None, gate: str, detail: str, *, check: str = "unmet") -> bool:
    """Record an unmet gate when the switch is on. True is not a pass."""
    from tools.gate_bypass import continue_past

    return continue_past(obj, root, gate, detail, check=check)


def _block_gate(block_id: str, code: int) -> tuple[str, str, str]:
    """Name the stop a block just hit. The check stays unmet or skipped."""
    if block_id == "B2":
        return "colmap_registration", f"B2 exited {code}", "unmet"
    if block_id == "B4":
        return "metres_per_unit", f"B4 exited {code}", "unmet"
    if block_id == "B5":
        return "pipeline_not_implemented", f"{block_id} is not implemented", "skipped"
    if code == 3:
        return "pipeline_not_implemented", f"{block_id} is not implemented", "skipped"
    return f"pipeline.{block_id}", f"{block_id} exited {code}", "unmet"


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
        caption = detector_caption(prompt_override)
    else:
        try:
            caption = detector_caption(caption_from_spec(spec))
        except ValueError:
            return []
    if not caption:
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


def ssh_key_path() -> Path:
    """The laptop key for the rented machine. It is not stored in the repo."""
    return Path.home() / ".ssh" / "ssh_sceneagentX_1"


def running_vast_host():
    """The one running Vast instance. A stopped host is not reused."""
    from tools.vast_host import VastError, list_instances

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
    key = ssh_key_path()
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
    key = ssh_key_path()
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


_REMOTE_ROOT = "/root/SceneagentX"
_MAMBA_ROOT = "/root/.local/share/mamba"
_NERF_PREFIX = (
    f"cd {_REMOTE_ROOT} && "
    f"export MAMBA_ROOT_PREFIX={_MAMBA_ROOT} && "
    f"export PATH={_MAMBA_ROOT}/envs/nerf/bin:{_MAMBA_ROOT}/bin:$PATH && "
)
_COPY_TIMEOUT = 6 * 60 * 60


class VastRunner:
    """Run one argv on the Vast host, inside the nerf env."""

    def __init__(self, host: VastHost, local_root: Path) -> None:
        self.host = host
        self.local_root = Path(local_root)
        self.copied_splat = False

    def __call__(self, cmd: Sequence[str], timeout: float) -> CmdResult:
        result = _ssh_command(self.host, _remote_shell(cmd, self.local_root), timeout)
        if result.returncode != 0:
            return result
        name = Path(str(cmd[0])).name if cmd else ""
        if name == "ns-process-data":
            orbit_dir = _flag_path(cmd, "--output-dir")
            if orbit_dir is not None:
                code = _pull_orbit(self.host, orbit_dir, self.local_root, splat=False)
                if code != 0:
                    return CmdResult(code, result.stdout, result.stderr + "sparse model was not copied back\n")
        elif name == "ns-train":
            orbit_dir = _flag_path(cmd, "--data")
            if orbit_dir is not None:
                code = _pull_orbit(self.host, orbit_dir, self.local_root, splat=True)
                if code != 0:
                    return CmdResult(code, result.stdout, result.stderr + "splat.ply was not copied back\n")
                self.copied_splat = True
        return result


def push_missing_raw_videos(host: VastHost, obj: str, root: Path | None = None) -> int:
    """Upload orbit videos that are not already on the Vast host."""
    try:
        name = _safe_token(obj)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    remote_dir = f"{_REMOTE_ROOT}/data/{name}/raw"
    for video in _orbit_videos(name, root):
        remote = f"{remote_dir}/{video.name}"
        exists, code = _remote_is_file(host, remote)
        if code != 0:
            return code
        if exists:
            continue
        made = _ssh_command(host, f"mkdir -p {shell_quote(remote_dir)}", 60)
        if made.returncode != 0:
            sys.stderr.write(made.stderr or "could not create the remote raw directory\n")
            return int(made.returncode)
        copied = _push_file(host, video, remote)
        if copied != 0:
            return copied
    return 0


def pull_reconstruction(host: VastHost, obj: str, root: Path | None = None, orbit: int = 1) -> int:
    """Copy splat.ply and the sparse model for one orbit back to the laptop."""
    try:
        name = _safe_token(obj)
        if orbit < 1:
            raise ValueError(f"refusing orbit {orbit}")
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    local_orbit = object_dir(name, root) / "colmap" / f"orbit{orbit}"
    return _pull_orbit(host, local_orbit, resolve_root(root), splat=True)


def _fail_b0_no_vast(obj: str, root: Path | None) -> int:
    begin_step(obj, "B0", root)
    message = "no running Vast instance"
    finish_step(obj, "B0", "fail", root, actual={"error": message})
    sys.stderr.write(message + "\n")
    return 1


def _orbit_videos(obj: str, root: Path | None) -> list[Path]:
    raw = raw_dir(obj, root)
    if not raw.is_dir():
        return []
    videos = []
    for path in sorted(raw.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path.stem.startswith("orbit"):
            videos.append(path)
    return videos


def _remote_shell(cmd: Sequence[str], local_root: Path) -> str:
    rewritten = [_rewrite_arg(str(part), local_root) for part in cmd]
    return _NERF_PREFIX + " ".join(shell_quote(part) for part in rewritten)


def _rewrite_arg(token: str, local_root: Path) -> str:
    path = Path(token)
    if not path.is_absolute():
        return token
    try:
        relative = path.resolve().relative_to(Path(local_root).resolve())
    except (OSError, ValueError):
        return token
    text = relative.as_posix()
    if text in {"", "."}:
        return _REMOTE_ROOT
    return f"{_REMOTE_ROOT}/{text}"


def _flag_path(cmd: Sequence[str], flag: str) -> Path | None:
    parts = [str(part) for part in cmd]
    if flag not in parts:
        return None
    index = parts.index(flag)
    if index + 1 >= len(parts):
        return None
    return Path(parts[index + 1])


def _ssh_argv(host: VastHost, remote_shell: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(ssh_key_path()),
        "-p",
        str(host.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "BatchMode=yes",
        f"root@{host.host}",
        remote_shell,
    ]


def _scp_argv(host: VastHost, source: str, dest: str, recursive: bool = False) -> list[str]:
    argv = [
        "scp",
        "-i",
        str(ssh_key_path()),
        "-P",
        str(host.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
    ]
    if recursive:
        argv.append("-r")
    argv.extend([source, dest])
    return argv


def _ssh_command(host: VastHost, remote_shell: str, timeout: float) -> CmdResult:
    key = ssh_key_path()
    if not key.is_file():
        return CmdResult(2, "", "ssh key is missing\n")
    try:
        done = subprocess.run(
            _ssh_argv(host, remote_shell),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CmdResult(127, "", str(exc))
    return CmdResult(int(done.returncode), done.stdout or "", done.stderr or "")


def _remote_is_file(host: VastHost, remote: str) -> tuple[bool, int]:
    done = _ssh_command(host, f"test -f {shell_quote(remote)}", 60)
    if done.returncode == 0:
        return True, 0
    if done.returncode == 1:
        return False, 0
    sys.stderr.write(done.stderr or "")
    return False, int(done.returncode)


def _push_file(host: VastHost, local: Path, remote: str) -> int:
    key = ssh_key_path()
    if not key.is_file():
        sys.stderr.write("ssh key is missing\n")
        return 2
    done = subprocess.run(
        _scp_argv(host, str(local), f"root@{host.host}:{remote}"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_COPY_TIMEOUT,
        check=False,
    )
    if done.returncode != 0:
        sys.stderr.write(done.stderr or "")
    return int(done.returncode)


def _pull_orbit(host: VastHost, local_orbit: Path, local_root: Path, *, splat: bool) -> int:
    remote_orbit = _rewrite_arg(str(local_orbit), local_root)
    if not remote_orbit.startswith(_REMOTE_ROOT):
        sys.stderr.write("refusing to copy a path outside the repo\n")
        return 2
    if splat:
        code = _pull_file(host, remote_orbit + "/splat.ply", local_orbit / "splat.ply")
        if code != 0:
            return code
    return _pull_tree(host, remote_orbit + "/sparse", local_orbit / "sparse")


def _pull_file(host: VastHost, remote: str, local: Path) -> int:
    return _pull_path(host, remote, local, recursive=False)


def _pull_tree(host: VastHost, remote: str, local: Path) -> int:
    return _pull_path(host, remote, local, recursive=True)


def _pull_path(host: VastHost, remote: str, local: Path, *, recursive: bool) -> int:
    key = ssh_key_path()
    if not key.is_file():
        sys.stderr.write("ssh key is missing\n")
        return 2
    local.parent.mkdir(parents=True, exist_ok=True)
    partial = local.with_name(local.name + ".partial")
    if partial.exists():
        if partial.is_dir():
            shutil.rmtree(partial)
        else:
            partial.unlink()
    done = subprocess.run(
        _scp_argv(host, f"root@{host.host}:{remote}", str(partial), recursive=recursive),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_COPY_TIMEOUT,
        check=False,
    )
    if done.returncode != 0:
        sys.stderr.write(done.stderr or "")
        if partial.is_dir():
            shutil.rmtree(partial)
        elif partial.is_file():
            partial.unlink()
        return int(done.returncode)
    if local.exists():
        if local.is_dir():
            shutil.rmtree(local)
        else:
            local.unlink()
    partial.replace(local)
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
