"""Block B job for any object. The same steps run for a new video as for the can.

B2 uses the sa env, which is where OpenCLIP is installed. A missing import is
installed there before crops are collected. A failed codebook stops the job
before the feature field. The nerf env is used for the splat projection and
the field, and it does not receive an OpenCLIP install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REFUSED = {"", ".", ".."}
FRAME_COVER = 0.90
NESTED_BOX_AREA = 0.9
OPEN_CLIP_MISSING = (
    "open_clip is not installed in this interpreter. "
    "Block B B2 runs in the sa env; install open_clip_torch there before collecting crops."
)


def require_object_name(name: str) -> str:
    """Refuse a path. A user name such as bottle is allowed."""
    text = str(name)
    if text != Path(text).name or text in REFUSED or "/" in text or "\\" in text:
        raise ValueError(f"refusing object name {name}")
    return text


def normalize_label(text: str) -> str:
    """Lower-case a detector label and drop punctuation."""
    cleaned = []
    for char in str(text).lower():
        cleaned.append(char if char.isalnum() or char.isspace() else " ")
    return " ".join("".join(cleaned).split())


def grounding_prompt(object_name: str) -> str:
    """One GroundingDINO phrase for the named target."""
    name = require_object_name(object_name)
    if name.endswith("."):
        return name
    return name + "."


def detector_caption(text: str) -> str:
    """First non-empty period-separated clause, with one trailing period.

    GroundingDINO treats each period as another phrase. A synonym list
    therefore becomes one phrase: the first clause only.
    """
    for part in str(text).split("."):
        clause = part.strip()
        if clause:
            return clause + "."
    return ""


def caption_from_spec(spec: dict) -> str:
    """Detector caption is the spec target, as one phrase. Other seen names are not added."""
    if not isinstance(spec, dict):
        raise ValueError("task spec must be an object")
    target = spec.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("task spec has no target")
    caption = detector_caption(grounding_prompt(target.strip()))
    if not caption:
        raise ValueError("task spec has no target")
    return caption


def target_label_indexes(names: list[str], target: str) -> list[int]:
    """Indexes whose label is the sister target and not another name."""
    wanted = normalize_label(target)
    if not wanted:
        raise ValueError("target name is empty")
    return [index for index, name in enumerate(names) if normalize_label(name) == wanted]


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Axis-aligned box around a mask, as exclusive xyxy. Empty masks have none."""
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _box_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _kept_box_indexes(masks: np.ndarray) -> list[int]:
    """Drop a box that contains a much smaller box. One box, and a lone large box, stay."""
    count = int(masks.shape[0])
    boxes = [_mask_box(masks[index]) for index in range(count)]
    areas = [0 if box is None else _box_area(box) for box in boxes]
    drop: set[int] = set()
    for outer in range(count):
        outer_box = boxes[outer]
        if outer_box is None:
            continue
        for inner in range(count):
            if inner == outer:
                continue
            inner_box = boxes[inner]
            if inner_box is None or areas[inner] <= 0 or areas[inner] >= areas[outer]:
                continue
            if areas[inner] >= NESTED_BOX_AREA * areas[outer]:
                continue
            if _box_contains(outer_box, inner_box):
                drop.add(outer)
                break
    return [index for index in range(count) if index not in drop]


def union_target_mask(masks: np.ndarray, names: list[str], target: str) -> tuple[np.ndarray, list[str]]:
    """OR boxes for the target only. An empty target keeps every box.

    After those target boxes are selected, a box that contains a smaller box
    of that same name is dropped when the smaller area is under 0.9 of the
    larger one. Other names are not selected. One remaining box is kept.
    """
    array = np.asarray(masks)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("masks must be (N, H, W)")
    labels = [str(name) for name in names]
    if labels and len(labels) != int(array.shape[0]):
        raise ValueError("each box needs one name")
    if not str(target).strip():
        if array.shape[0] == 0:
            return np.zeros((0, 0), dtype=bool), []
        return np.any(array.astype(bool), axis=0), labels
    indexes = target_label_indexes(labels, target)
    if not indexes:
        return np.zeros(array.shape[1:], dtype=bool), []
    chosen = array[indexes].astype(bool)
    keep = _kept_box_indexes(chosen)
    if not keep:
        return np.zeros(array.shape[1:], dtype=bool), []
    kept = chosen[keep]
    return np.any(kept, axis=0), [labels[indexes[index]] for index in keep]


def mask_covers_frame(foreground_pixels: int, frame_pixels: int, limit: float = FRAME_COVER) -> bool:
    """True when a mask covers about 90 percent of the frame."""
    if frame_pixels <= 0:
        return False
    return foreground_pixels / frame_pixels >= limit


def feature_field_may_start(
    job: str | None,
    *,
    view_fractions: list[float] | None = None,
    covers_room: bool = False,
    span_ratio: float | None = None,
) -> bool:
    """A frame-covering view is dropped. Half-scene refusal applies to a room job.

    An orbit job skips the half-scene test. Views that cover the frame are
    still dropped, so a field is not started from those views alone.
    """
    fractions = [float(item) for item in (view_fractions or [])]
    if fractions and all(item >= FRAME_COVER for item in fractions):
        return False
    if job == "orbit":
        return True
    if covers_room or (span_ratio is not None and span_ratio >= 0.5):
        return False
    return True


def codebook_collapsed(max_abs: float, ae_loss: float) -> bool:
    """True when the autoencoder wrote a zero latent or stopped learning."""
    return float(max_abs) == 0.0 or float(ae_loss) >= 0.99


def shell_quote(text: str) -> str:
    return "'" + str(text).replace("'", "'\\''") + "'"


def write_host_text(stream, text: str) -> None:
    """Write host output on a Windows console. A checkmark must not raise."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    stream.write(safe)


def _orbit_name(name: str) -> str:
    text = str(name)
    if text != Path(text).name or not text.startswith("orbit") or "/" in text or "\\" in text:
        raise ValueError(f"refusing orbit name {name}")
    return text


def _config_path(path: str) -> str:
    text = str(path)
    if not text.startswith("/") or any(char in text for char in " \t\n\r'\"\\"):
        raise ValueError(f"refusing splat config {path}")
    return text


def host_launch_script(
    object_name: str,
    orbits: list[tuple[str, str]],
    *,
    prompt: str | None = None,
    spec: dict | None = None,
    box_threshold: float | None = None,
    root: str = "/root/SceneagentX",
    log_stem: str | None = None,
    skip_flag: str | None = None,
) -> str:
    """Bash that backs up, then runs B1–B4. B2 failure returns before the feature field.

    A task spec or an explicit prompt supplies one detector phrase. The folder
    name is not used as that caption when the spec names a target.
    """
    name = require_object_name(object_name)
    if not orbits:
        raise ValueError("Block B needs one frozen splat config per orbit")
    target = None
    if spec is not None:
        caption = detector_caption(caption_from_spec(spec))
        target = str(spec["target"]).strip()
    elif prompt is None:
        caption = grounding_prompt(name)
    else:
        caption = detector_caption(str(prompt))
    if not caption.strip() or "\n" in caption or "\r" in caption:
        raise ValueError("prompt must be one non-empty line")
    if box_threshold is not None:
        box_threshold = float(box_threshold)
        if not 0.0 <= box_threshold <= 1.0:
            raise ValueError("box_threshold must be between 0 and 1")
    stem = log_stem or f"{name}_block_b"
    if stem != Path(stem).name:
        raise ValueError(f"refusing log name {log_stem}")
    root_path = str(root).rstrip("/")
    pairs = [(_orbit_name(orbit), _config_path(config)) for orbit, config in orbits]
    job = _job_body(name, pairs, caption, root_path, skip_flag, target, box_threshold)
    script_path = f"/root/logs/{stem}.sh"
    pid_path = f"/root/logs/{stem}.pid"
    log_path = f"/root/logs/{stem}.log"
    return "\n".join(
        [
            "set -eu",
            "mkdir -p /root/logs",
            f"cat > {script_path} << 'ENDJOB'",
            job,
            "ENDJOB",
            f"chmod +x {script_path}",
            f"if [ -f {pid_path} ]; then",
            f"  old=$(cat {pid_path})",
            '  if [ -d "/proc/$old" ]; then',
            '    echo ALREADY_RUNNING "$old"',
            "    exit 3",
            "  fi",
            "fi",
            f"cd {root_path}",
            f"nohup bash {script_path} > {log_path} 2>&1 &",
            f"echo $! > {pid_path}",
            f"echo STARTED $(cat {pid_path})",
            "",
        ]
    )


def _job_body(
    name: str,
    orbits: list[tuple[str, str]],
    caption: str,
    root: str,
    skip_flag: str | None,
    target: str | None = None,
    box_threshold: float | None = None,
) -> str:
    quoted_prompt = shell_quote(caption)
    detect = f'"$GSAM" -u vast-cluster/scripts/object_semantics.py --obj {name} --prompt "$PROMPT"'
    if target:
        detect += ' --target "$TARGET"'
    if box_threshold is not None:
        detect += f" --box-threshold {box_threshold}"
    detect += ' --orbit "$orbit" --root "$ROOT"'
    b1_loop = "\n".join(
        [
            "  for orbit in " + " ".join(orbit for orbit, _config in orbits) + "; do",
            '    mark "B1 $orbit"',
            f"    {detect}",
            '    mark "B1_EXIT $orbit $?"',
            "  done",
        ]
    )
    if skip_flag:
        flag = _config_path(skip_flag) if skip_flag.startswith("/") else skip_flag
        if not str(flag).startswith("/root/logs/"):
            raise ValueError(f"refusing skip flag {skip_flag}")
        b1 = "\n".join(
            [
                f"if [ -f {flag} ]; then",
                f"  rm -f {flag}",
                "  mark B1_SKIP",
                "else",
                b1_loop,
                "fi",
            ]
        )
    else:
        b1 = b1_loop
    projects = []
    for orbit, config in orbits:
        projects.extend(
            [
                f"run_orbit {orbit} {config}",
            ]
        )
    clusters = []
    for orbit, _config in orbits:
        script = "object_langsplat_b4.py" if orbit == "orbit1" else "object_cluster_can_orbit.py"
        clusters.extend(
            [
                f'mark "B4 {orbit}"',
                f'"$NERF" -u vast-cluster/scripts/{script} --obj {name} --root "$ROOT" --orbit {orbit}',
                f'mark "B4_EXIT {orbit} $?"',
                "",
            ]
        )
    summary_orbits = ", ".join(repr(orbit) for orbit, _config in orbits)
    from tools.gate_bypass import enabled

    bypass = enabled()
    backup_fail = "echo BACKUP_UNMET" if bypass else "echo BACKUP_FAIL json; exit 4"
    orbit_backup_fail = 'echo BACKUP_UNMET "$orbit"' if bypass else 'echo BACKUP_FAIL "$orbit"; exit 4'
    codebook_backup_fail = 'echo BACKUP_UNMET "$codebook_name"' if bypass else 'echo BACKUP_FAIL "$codebook_name"; exit 4'
    semantics_backup_fail = "echo BACKUP_UNMET semantics" if bypass else "echo BACKUP_FAIL semantics; exit 4"
    if bypass:
        b2_stop = "\n".join(
            [
                'if [ "$b2_rc" -ne 0 ]; then',
                '  mark "B2_UNMET codebook"',
                "fi",
            ]
        )
    else:
        b2_stop = "\n".join(
            [
                'if [ "$b2_rc" -ne 0 ]; then',
                "  mark STOP_AFTER_B2",
                '  exit "$b2_rc"',
                "fi",
            ]
        )
    lines = [
        "set -u",
        "export MAMBA_ROOT_PREFIX=/root/.local/share/mamba",
        "export PYTHONUNBUFFERED=1",
        *(["export SCENEAGENT_BYPASS_GATES=1"] if bypass else []),
        f"ROOT={root}",
        'cd "$ROOT"',
        'GSAM="$MAMBA_ROOT_PREFIX/envs/gsam2/bin/python"',
        'NERF="$MAMBA_ROOT_PREFIX/envs/nerf/bin/python"',
        'SA="$MAMBA_ROOT_PREFIX/envs/sa/bin/python"',
        f"PROMPT={quoted_prompt}",
        *([f"TARGET={shell_quote(target)}"] if target else []),
        'mark() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }',
        "",
        f'DEST="$ROOT/data/{name}/ingest/block_b_first"',
        'if [ ! -f "$DEST/KEEP" ]; then',
        '  mkdir -p "$DEST/ingest" "$DEST/masks" "$DEST/semantics"',
        f'  cp -a "$ROOT"/data/{name}/ingest/block_b*.json "$DEST/ingest/" || {{ {backup_fail}; }}',
        "  for orbit in " + " ".join(orbit for orbit, _config in orbits) + "; do",
        f'    if [ -d "$ROOT/data/{name}/masks/$orbit" ]; then',
        f'      cp -a "$ROOT/data/{name}/masks/$orbit" "$DEST/masks/" || {{ {orbit_backup_fail}; }}',
        "    fi",
        "  done",
        "  for codebook_name in codebook.json codebook_orbit2.json codebook_orbit3.json; do",
        f'    if [ -f "$ROOT/data/{name}/$codebook_name" ]; then',
        f'      cp -a "$ROOT/data/{name}/$codebook_name" "$DEST/" || {{ {codebook_backup_fail}; }}',
        "    fi",
        "  done",
        f'  if [ -d "$ROOT/data/{name}/semantics" ]; then',
        f'    cp -a "$ROOT/data/{name}/semantics/." "$DEST/semantics/" || {{ {semantics_backup_fail}; }}',
        "  fi",
        '  date -u > "$DEST/KEEP"',
        "  mark BACKUP_OK",
        '  du -sh "$DEST" || true',
        "else",
        "  mark BACKUP_ALREADY",
        "fi",
        "",
        'if [ ! -x "$GSAM" ]; then echo gsam2_python_missing; exit 2; fi',
        'if [ ! -x "$NERF" ]; then echo nerf_python_missing; exit 2; fi',
        'if [ ! -x "$SA" ]; then echo sa_python_missing; exit 2; fi',
        "MM=micromamba",
        "if ! command -v micromamba >/dev/null 2>&1; then MM=/usr/local/bin/micromamba; fi",
        'if ! "$SA" -c "import open_clip"; then',
        "  mark OPEN_CLIP_INSTALL",
        '  "$MM" run -n sa pip install open_clip_torch',
        "fi",
        '"$SA" -c "import open_clip" || { echo open_clip_missing; exit 2; }',
        "",
        b1,
        "",
        "mark B2",
        f'"$SA" -u vast-cluster/scripts/object_langsplat_b2.py --obj {name} --root "$ROOT"',
        "b2_rc=$?",
        'mark "B2_EXIT $b2_rc"',
        b2_stop,
        "",
        "run_orbit() {",
        '  orbit="$1"',
        '  cfg="$2"',
        '  mark "PROJECT $orbit"',
        f'  "$NERF" -u vast-cluster/scripts/object_crop_iou.py --obj {name} --prompt "$PROMPT" --orbit "$orbit" --config "$cfg" --root "$ROOT"',
        '  mark "PROJECT_EXIT $orbit $?"',
        '  mark "FIELD $orbit"',
        f'  "$NERF" -u vast-cluster/scripts/object_langsplat_b3.py --obj {name} --root "$ROOT" --orbit "$orbit"',
        '  mark "FIELD_EXIT $orbit $?"',
        "}",
        "",
        *projects,
        "",
        *clusters,
        '"$NERF" - << \'PY\'',
        "import json",
        "from pathlib import Path",
        f'root = Path("{root}/data/{name}")',
        f"for orbit in ({summary_orbits}):",
        '    obj_path = root / "masks" / orbit / "object.json"',
        '    obj = json.loads(obj_path.read_text(encoding="utf-8")) if obj_path.is_file() else {}',
        '    npy = root / "masks" / orbit / "gaussian_in_mask.npy"',
        '    if orbit == "orbit1":',
        '        feat = root / "semantics" / "gaussian_features.npy"',
        '        inst = root / "semantics" / "orbit1" / "instances.json"',
        "    else:",
        '        feat = root / "semantics" / orbit / "gaussian_features.npy"',
        '        inst = root / "semantics" / orbit / "instances.json"',
        "    print(",
        '        "SUMMARY", orbit, "det", obj.get("n_views_with_detection"), "/", obj.get("n_views"),',
        '        "largest", obj.get("largest_foreground_pixels"), "inside", obj.get("n_inside"),',
        '        "covers", obj.get("covers_room"), "span", obj.get("span_inside"), obj.get("span_scene"),',
        '        "projected", obj.get("projected"), "npy", npy.is_file(), "feat", feat.is_file(),',
        '        "inst", inst.is_file(), flush=True,',
        "    )",
        "PY",
        "mark JOB_DONE",
    ]
    return "\n".join(lines)
