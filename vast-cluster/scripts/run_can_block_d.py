#!/usr/bin/env python3
"""Copy Block D writers to the live Vast host and run GPU encode / ply concat.

Does not train a prior. Does not invent mass. start_d2 is not called here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = Path.home() / ".ssh" / "ssh_sceneagentX_1"
VASTAI = ROOT / ".venv" / "Scripts" / "vastai.exe"
SA = "/root/.local/share/mamba/envs/sa/bin/python"
REMOTE = "/root/SceneagentX"

COPIES = (
    (ROOT / "vast-cluster" / "scripts" / "run_bottle_codebook.py", f"{REMOTE}/vast-cluster/scripts/run_bottle_codebook.py"),
    (ROOT / "vast-cluster" / "scripts" / "isaac_drop_slide.py", f"{REMOTE}/vast-cluster/scripts/isaac_drop_slide.py"),
)


def resolve_host() -> tuple[str, int]:
    listed = subprocess.run([str(VASTAI), "show", "instances", "--raw"], capture_output=True, text=True, check=False)
    rows = json.loads(listed.stdout or "[]")
    running = [row for row in rows if row.get("actual_status") == "running"]
    if len(running) != 1:
        raise SystemExit(f"expected one running instance, found {[row.get('id') for row in running]}")
    host = str(running[0].get("public_ipaddr") or "")
    port = int(running[0].get("direct_port_start") or 0)
    if not host or not port:
        raise SystemExit("running instance has no public IP/port")
    print(f"VAST id={running[0].get('id')} ip={host} port={port}", flush=True)
    return host, port


def ssh(host: str, port: int, script: str, timeout: int = 120) -> tuple[int, str, str]:
    done = subprocess.run(
        [
            "ssh",
            "-i",
            str(KEY),
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "BatchMode=yes",
            f"root@{host}",
            "bash",
            "-s",
        ],
        input=script.replace("\r", "").encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return done.returncode, (done.stdout or b"").decode("utf-8", "replace"), (done.stderr or b"").decode("utf-8", "replace")


def scp_to(host: str, port: int, local: Path, remote: str) -> int:
    done = subprocess.run(
        [
            "scp",
            "-i",
            str(KEY),
            "-P",
            str(port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            str(local),
            f"root@{host}:{remote}",
        ],
        capture_output=True,
        check=False,
    )
    return done.returncode


def scp_from(host: str, port: int, remote: str, local: Path) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [
            "scp",
            "-i",
            str(KEY),
            "-P",
            str(port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            f"root@{host}:{remote}",
            str(local),
        ],
        capture_output=True,
        check=False,
    )
    return done.returncode


def main() -> int:
    host, port = resolve_host()
    probe_code, probe_out, probe_err = ssh(
        host,
        port,
        """
set -eu
echo PLY_ORBIT1 $(test -f /root/SceneagentX/data/can/colmap/orbit1/splat.ply && echo yes || echo no)
echo PLY_FIND
find /root/SceneagentX -name splat.ply 2>/dev/null | head
echo TOOLS
ls /root/SceneagentX/tools | head
echo SA $(test -x /root/.local/share/mamba/envs/sa/bin/python && echo yes || echo no)
echo MASK $(test -f /root/SceneagentX/data/can/masks/orbit1/gaussian_in_mask.npy && echo yes || echo no)
echo IOU $(test -f /root/SceneagentX/data/can/iou/orbit1/frame_00167.png && echo yes || echo no)
""",
        timeout=45,
    )
    sys.stdout.write(probe_out)
    sys.stderr.write(probe_err)
    if probe_code != 0:
        return probe_code
    tools_code = subprocess.run(
        [
            "scp",
            "-i",
            str(KEY),
            "-P",
            str(port),
            "-r",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            str(ROOT / "tools"),
            f"root@{host}:{REMOTE}/",
        ],
        capture_output=True,
        check=False,
    ).returncode
    print(f"SCP tools/ exit={tools_code}", flush=True)
    if tools_code != 0:
        return tools_code
    for local, remote in COPIES:
        code = scp_to(host, port, local, remote)
        print(f"SCP {local.name} -> {remote} exit={code}", flush=True)
        if code != 0:
            return code
    encode_code, encode_out, encode_err = ssh(
        host,
        port,
        f"""
set -eu
export PYTHONPATH={REMOTE}
{SA} {REMOTE}/vast-cluster/scripts/run_bottle_codebook.py --obj can --orbit orbit1 --root {REMOTE}
echo ENCODE_ORBIT1_EXIT:$?
{SA} - <<'PY'
from pathlib import Path
import json
import numpy as np
from tools.block_d import write_prior_input
from tools.step02_splatfacto import splat_path

root = Path("{REMOTE}")
ply = splat_path("can", 1, root)
npz = root / "data" / "can" / "codebook_gaussians.npz"
vectors = None
if npz.is_file():
    loaded = np.load(npz)
    assigned = loaded["per_gaussian"]
    if not np.isnan(assigned).any():
        vectors = assigned
print("PLY", ply, ply.is_file() if ply else False)
print("NPZ", npz.is_file(), "assigned", None if vectors is None else vectors.shape)
status = write_prior_input("can", root, ply=ply if ply and ply.is_file() else None, vectors=vectors)
print("PRIOR_INPUT", status)
codebook = json.loads((root / "data" / "can" / "codebook.json").read_text())
print("SCENE_MEAN_MEASURED", codebook.get("scene_mean_measured"))
print("LEARNED_PRIOR", codebook.get("learned_prior"))
PY
""",
        timeout=1800,
    )
    sys.stdout.write(encode_out)
    sys.stderr.write(encode_err)
    print(f"ENCODE_SSH_EXIT {encode_code}", flush=True)
    isaac_code, isaac_out, isaac_err = ssh(
        host,
        port,
        f"export PYTHONPATH={REMOTE}; {SA} {REMOTE}/vast-cluster/scripts/isaac_drop_slide.py --obj can --root {REMOTE} --timeout 60",
        timeout=90,
    )
    sys.stdout.write(isaac_out)
    sys.stderr.write(isaac_err)
    print(f"ISAAC_SSH_EXIT {isaac_code}", flush=True)
    pulls = (
        ("data/can/codebook.json", ROOT / "data" / "can" / "codebook.json"),
        ("data/can/codebook_gaussians.npz", ROOT / "data" / "can" / "codebook_gaussians.npz"),
        ("data/can/semantics/gaussian_feature_idx.npy", ROOT / "data" / "can" / "semantics" / "gaussian_feature_idx.npy"),
        ("data/can/semantics/prior_input.npy", ROOT / "data" / "can" / "semantics" / "prior_input.npy"),
        ("data/can/ingest/isaac_drop_slide.json", ROOT / "data" / "can" / "ingest" / "isaac_drop_slide.json"),
    )
    for remote_rel, local in pulls:
        code = scp_from(host, port, f"{REMOTE}/{remote_rel}", local)
        print(f"PULL {remote_rel} exit={code} exists={local.is_file()}", flush=True)
    return 0 if encode_code == 0 else encode_code


if __name__ == "__main__":
    raise SystemExit(main())
