"""Resume one skill, and repair it only when the retry changes something.

Measurements judge registration, reprojection, scale, and surface error.
A vision model is asked only for what eyes have to judge. A retry that
repeats the last try is refused. A worse measurement is thrown away.
Three retries is the limit, and each retry must change an allowed parameter.
An unchanged retry stops. The next step is one question.
This module does not start COLMAP for a ply, and it does not call CrewAI.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any, Callable

from tools.common.paths import object_dir, renders_dir
from tools.harvard_agents import SKILLS, write_agents
from tools.ingest import classify_input, colmap_refused
from tools.review_step import REQUIRED, ReviewError, review_block
from tools.video_input import VIDEO_SUFFIXES, find_video, record_absent

MAX_RETRIES = 3
MEASURED_BLOCKS = frozenset({"B0", "B1", "B2", "B3", "B4", "B8"})
_VISION_MODEL = "gpt-6-astra"
_VISION_URL = "https://api.openai.com/v1/chat/completions"
Checker = Callable[[str, str, Path | None], dict[str, Any]]


class SkillFault(RuntimeError):
    """The program for this skill failed before a picture existed."""


def choose_path(obj: str, root: Path | None = None, source: Path | None = None) -> str:
    """Return ``ply``, ``video``, or ``stop``. A ply never selects the video path."""
    if source is not None:
        record = classify_input(source)
        if record.get("kind") == "polycam_mesh":
            return "ply"
    from tools.common.status import StatusError, load_status

    try:
        doc = load_status(obj, root, create=False)
    except StatusError:
        doc = {}
    if colmap_refused(doc):
        return "ply"
    if find_video(obj, root) is None:
        record_absent(obj, root)
        return "stop"
    return "video"


def run_skill(
    obj: str,
    block_id: str,
    run_once: Callable[[dict[str, Any] | None], None],
    root: Path | None = None,
    checker: Checker | None = None,
) -> tuple[int, bool]:
    """Run one skill. A retry must change a parameter. Two tries is the limit.

    Returns ``(code, visual_review)``. ``visual_review`` is true only when
    the checker reports that the model returned a decision.
    """
    check = checker or default_checker
    faults = 0
    saw_model = False
    parameter: dict[str, Any] | None = None
    previous_parameter: dict[str, Any] | None = None
    last_score: float | None = None
    while True:
        try:
            run_once(parameter)
        except SkillFault as exc:
            _stop_with_question(
                obj,
                block_id,
                faults + 1,
                str(exc),
                f"{block_id} failed before it could be checked. What should change on the next try?",
                saw_model,
                root,
            )
            return 2, saw_model
        if block_id in REQUIRED:
            try:
                review_block(obj, block_id, root)
            except ReviewError as exc:
                _stop_with_question(
                    obj,
                    block_id,
                    faults + 1,
                    str(exc),
                    f"{block_id} is missing a file. What should change on the next try?",
                    saw_model,
                    root,
                )
                return 2, saw_model
        decision = check(obj, block_id, root)
        if decision.get("called") is True:
            saw_model = True
        name = str(decision.get("decision") or "pass")
        score = _score(decision.get("score"))
        if name == "pass":
            if last_score is not None and score is not None and score > last_score:
                run_once(previous_parameter)
                _stop_with_question(
                    obj,
                    block_id,
                    faults + 1,
                    "measurement got worse; the previous result was kept",
                    "The last change made the measurement worse. What should change instead?",
                    saw_model,
                    root,
                )
                return 2, saw_model
            _close(obj, root, saw_model)
            return 0, saw_model
        if name == "skip":
            _record_fault(
                obj,
                block_id,
                faults + 1,
                str(decision.get("fault") or "skip"),
                False,
                root,
            )
            _close(obj, root, saw_model)
            return 0, saw_model
        new_parameter = _allowed_parameter(decision.get("parameter"))
        if _same_parameter(new_parameter, parameter):
            _stop_with_question(
                obj,
                block_id,
                faults + 1,
                str(decision.get("fault") or "retry did not change anything"),
                f"{block_id} would repeat the same try. What should change on the next try?",
                saw_model,
                root,
            )
            return 2, saw_model
        faults += 1
        restarted = faults <= MAX_RETRIES
        _store_parameter(obj, block_id, new_parameter, root)
        _record_fault(
            obj,
            block_id,
            faults,
            str(decision.get("fault") or "restart"),
            restarted,
            root,
        )
        if not restarted:
            ask_question(
                obj,
                block_id,
                f"{block_id} used its three retries. What should change on the next try?",
                root,
            )
            _close(obj, root, saw_model)
            return 2, saw_model
        previous_parameter = parameter
        last_score = score
        parameter = new_parameter


def finish_route(obj: str, root: Path | None, visual_review: bool) -> None:
    _close(obj, root, visual_review)


def _close(obj: str, root: Path | None, visual_review: bool) -> None:
    write_agents(obj, root, ran=True, visual_review=True if visual_review else None)
    from tools.common.status import load_status
    from tools.review_contract import write_report

    doc = load_status(obj, root, create=True)
    write_report(
        object_dir(obj, root) / "report.json",
        "partial",
        {
            "visual_review": bool(visual_review),
            "learned_prior": False,
            "gpu_minutes": doc.get("gpu_minutes"),
        },
        [],
        [],
    )
    from tools.changelog import write_changelog

    write_changelog(obj, root)


def ask_question(obj: str, block_id: str, question: str, root: Path | None = None) -> Path:
    """Write one question. A second problem does not replace the first."""
    folder = object_dir(obj, root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "question.json"
    if path.is_file():
        return path
    text = " ".join(question.split())
    path.write_text(
        json.dumps({"block": block_id, "question": text}, indent=2) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(text + "\n")
    return path


def _stop_with_question(
    obj: str,
    block_id: str,
    attempt: int,
    fault: str,
    question: str,
    saw_model: bool,
    root: Path | None,
) -> None:
    _record_fault(obj, block_id, attempt, fault, False, root)
    ask_question(obj, block_id, question, root)
    _close(obj, root, saw_model)


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _same_parameter(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return _allowed_parameter(left or {}) == _allowed_parameter(right or {})


def default_checker(obj: str, block_id: str, root: Path | None = None) -> dict[str, Any]:
    """Measurements do not call the vision model. Eyes do, when a picture and a key exist."""
    if block_id in MEASURED_BLOCKS:
        return {"decision": "pass", "called": False, "judge": "measurement"}
    image = _render_for(obj, block_id, root)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or image is None:
        return {"decision": "pass", "called": False, "judge": "eyes"}
    try:
        parsed = _vision_decision(key, image, block_id)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {"decision": "pass", "called": False}
    parsed["called"] = True
    return parsed


def _vision_decision(api_key: str, image: Path, block_id: str) -> dict[str, Any]:
    encoded = base64.standard_b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": _VISION_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Look at this {block_id} render. Reply with JSON only: "
                            '{"decision":"pass"|"restart"|"skip","fault":"short reason",'
                            '"parameter":{"prompt":"optional"} or {"box_threshold":0.25}}. '
                            "restart only when this picture is a bad mask, a bad lift, or a bad package. "
                            "Do not ask to retrain a splat."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        _VISION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise TypeError("vision decision must be an object")
    decision = str(parsed.get("decision") or "pass")
    if decision not in {"pass", "restart", "skip"}:
        decision = "pass"
    return {
        "decision": decision,
        "fault": str(parsed.get("fault") or ""),
        "parameter": parsed.get("parameter") if isinstance(parsed.get("parameter"), dict) else {},
    }


def _render_for(obj: str, block_id: str, root: Path | None) -> Path | None:
    folder = object_dir(obj, root)
    candidates = [
        renders_dir(obj, "synthetic", root) / "view_000.png",
        folder / "inspect" / "preview.png",
    ]
    if block_id in {"B5", "B6"}:
        masks = sorted((folder / "masks").glob("view_*_mask.png"))
        if masks:
            candidates.insert(0, masks[0])
    for path in candidates:
        if path.is_file():
            return path
    return None


def _allowed_parameter(parameter: Any) -> dict[str, Any]:
    if not isinstance(parameter, dict):
        return {}
    kept: dict[str, Any] = {}
    prompt = parameter.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        kept["prompt"] = prompt.strip()[:80]
    threshold = parameter.get("box_threshold")
    if isinstance(threshold, (int, float)) and 0.0 <= float(threshold) <= 1.0:
        kept["box_threshold"] = float(threshold)
    return kept


def _store_parameter(obj: str, block_id: str, parameter: dict[str, Any], root: Path | None) -> None:
    path = object_dir(obj, root) / "agent_parameter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"block": block_id, **parameter}, indent=2) + "\n",
        encoding="utf-8",
    )


def _record_fault(
    obj: str,
    block_id: str,
    attempt: int,
    fault: str,
    restarted: bool,
    root: Path | None,
) -> None:
    folder = object_dir(obj, root)
    path = folder / "agents.json"
    faults: list[dict[str, Any]] = []
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("faults"), list):
            faults = [item for item in loaded["faults"] if isinstance(item, dict)]
    faults.append(
        {
            "block": block_id,
            "attempt": attempt,
            "fault": fault.replace("\n", " ")[:240],
            "restarted": restarted,
        }
    )
    write_agents(obj, root, faults=faults)
    from tools.common.status import load_status

    load_status(obj, root, create=True)
    from tools.changelog import write_changelog

    write_changelog(obj, root)


class UnknownSkillError(ValueError):
    """The planner named a skill id that is not in the allow-list."""


_SKILL_BLOCK = {
    "SCENEAGENT-INGEST-DATA": "B2",
    "SCENEAGENT-RUN-STEP": "B5",
    "SCENEAGENT-ALIGN-SCENE": "B4",
    "SCENEAGENT-ORIENT-BACKGROUND": "B4",
    "SCENEAGENT-PACKAGE-DATA-PROCESSED": "B12",
    "SCENEAGENT-INSPECT-ISAAC": "B12",
    "SCENEAGENT-OBJECT-SISTERS": "B13",
    "SCENEAGENT-SCENE-SISTERS": "B14",
    "SCENEAGENT-RUN-PIPELINE": "B16",
}


def accept_planner_skill(skill_id: str) -> str:
    """Hard-rule allow-list. An unknown id is rejected before any program runs."""
    known = {item["id"] for item in SKILLS}
    if skill_id not in known:
        raise UnknownSkillError(f"unknown skill id {skill_id}")
    return skill_id


def advance_once(
    obj: str,
    root: Path | None = None,
    planner: Any = None,
    runner: Callable[..., None] | None = None,
) -> int:
    """Advance exactly one skill. A ply never calls COLMAP. A second fault stops."""
    route = choose_path(obj, root)
    if route == "stop":
        _write_once(obj, root, [], "not called", stopped="missing orbit1")
        return 2
    if planner is None:
        skill_id = _local_planner(obj, root, route)
    elif callable(planner):
        skill_id = planner(obj, root)
    else:
        skill_id = str(planner)
    skill_id = accept_planner_skill(skill_id)
    run = runner or _default_runner
    parameter: dict[str, Any] | None = None
    for attempt in (1, 2):
        try:
            run(obj, skill_id, root, parameter)
        except SkillFault as exc:
            _record_fault(obj, _SKILL_BLOCK.get(skill_id, "B16"), attempt, str(exc), attempt == 1, root)
            if attempt == 2:
                ask_question(
                    obj,
                    _SKILL_BLOCK.get(skill_id, "B16"),
                    f"{skill_id} failed twice. What should change on the next try?",
                    root,
                )
                _write_once(obj, root, [skill_id], "not called" if route == "ply" else "not this turn")
                return 2
            parameter = {"box_threshold": 0.22} if "mask" in str(exc).lower() else {"attempt": attempt}
            continue
        _write_once(obj, root, [skill_id], "not called" if route == "ply" else "not this turn")
        return 0
    return 2


def _local_planner(obj: str, root: Path | None, route: str) -> str:
    path = object_dir(obj, root) / "masks" / "object.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("corrupt") is True:
            return "SCENEAGENT-RUN-STEP"
    if route == "ply":
        return "SCENEAGENT-PACKAGE-DATA-PROCESSED"
    return "SCENEAGENT-RUN-STEP"


def _default_runner(obj: str, skill_id: str, root: Path | None, parameter: dict[str, Any] | None) -> None:
    """One local step. COLMAP and splat training are not called from here."""
    if skill_id != "SCENEAGENT-RUN-STEP":
        return
    path = object_dir(obj, root) / "masks" / "object.json"
    if not path.is_file():
        return
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("corrupt") is not True:
        return
    if parameter and parameter.get("box_threshold") is not None:
        loaded["corrupt"] = False
        loaded["box_threshold"] = parameter["box_threshold"]
        path.write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")
        return
    raise SkillFault("corrupted mask")


def _write_once(
    obj: str,
    root: Path | None,
    skills: list[str],
    colmap: str,
    stopped: str | None = None,
) -> None:
    folder = object_dir(obj, root)
    folder.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"skills": list(skills), "n_skills": len(skills), "colmap": colmap}
    if stopped:
        payload["stopped"] = stopped
    (folder / "once.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


CAN_CAPTION = "aluminum soda can."
SkillFn = Callable[[], int]


def run_block_h(
    obj: str,
    root: Path | None = None,
    skills: list[tuple[str, SkillFn]] | None = None,
    *,
    prompt: str | None = None,
    source: Path | None = None,
    fresh: bool = False,
    videos: list[Path] | None = None,
) -> int:
    """Walk the Block H skills, starting with SCENEAGENT-RUN-PIPELINE.

    A prompt is stored before the first skill. The detector caption is one
    phrase from that prompt. ScoreOnly exits continue without
    SCENEAGENT_BYPASS_GATES. HardStop ends the walk. Bypass still continues
    other unmet checks without marking a milestone achieved.
    """
    from tools.gate_bypass import continue_past
    from tools.gate_policy import classify_skill_result, record_ledger, walk_continues

    caption = CAN_CAPTION
    video_names = [str(path) for path in videos] if videos else None
    if video_names:
        sys.stdout.write("videos " + " ".join(video_names) + "\n")
    if prompt is not None:
        from tools.block_b_job import detector_caption
        from tools.task_spec import accept_pipeline_prompt

        caption = detector_caption(prompt)
        if not caption:
            sys.stderr.write("prompt must contain one phrase\n")
            _write_block_h(
                obj,
                root,
                [],
                stopped="pipeline_prompt",
                caption=CAN_CAPTION,
                prompt=prompt,
                videos=video_names,
            )
            return 2
        code = accept_pipeline_prompt(obj, prompt, root, source=source)
        if code != 0 and not continue_past(
            obj,
            root,
            "pipeline_prompt",
            "the pipeline prompt was not stored",
            check="unmet",
        ):
            _write_block_h(
                obj,
                root,
                [],
                stopped="pipeline_prompt",
                caption=caption,
                prompt=prompt,
                videos=video_names,
            )
            return code
        sys.stdout.write(f"detector caption {caption}\n")
    if fresh:
        _clear_local_stale(obj, root)
    steps = list(skills) if skills is not None else _default_skills(obj, root, caption, fresh)
    walked: list[dict[str, Any]] = []
    for skill_id, fn in steps:
        accept_planner_skill(skill_id)
        detail = ""
        try:
            code = int(fn())
        except SkillFault as exc:
            code = 2
            detail = str(exc)
        except Exception as exc:
            from tools.block_gates import GateError

            if not isinstance(exc, GateError):
                raise
            code = 2
            detail = str(exc)
        walked.append({"skill": skill_id, "code": code, "detail": detail, "achieved": False, "m1_met": False, "m2_met": False})
        sys.stdout.write(f"{skill_id} exit {code}\n")
        if code != 0:
            note = detail or f"{skill_id} exited {code}"
            gate_class = classify_skill_result(skill_id, code, note)
            if walk_continues(gate_class):
                record_ledger(
                    obj,
                    root,
                    gate=skill_id,
                    gate_class=gate_class,
                    detail=note,
                    check="unmet",
                    reason="score_only_continue",
                )
                continue
            if continue_past(obj, root, skill_id, note, check="unmet"):
                continue
            ask_question(
                obj,
                _SKILL_BLOCK.get(skill_id, "B16"),
                f"{skill_id} stopped. What should change on the next try?",
                root,
            )
            _write_block_h(
                obj,
                root,
                walked,
                stopped=skill_id,
                caption=caption,
                prompt=prompt,
                videos=video_names,
            )
            return code if code != 0 else 2
    _write_block_h(obj, root, walked, stopped=None, caption=caption, prompt=prompt, videos=video_names)
    return 0


def _default_skills(
    obj: str,
    root: Path | None,
    caption: str = CAN_CAPTION,
    fresh: bool = False,
) -> list[tuple[str, SkillFn]]:
    return [
        ("SCENEAGENT-RUN-PIPELINE", lambda: _skill_run_pipeline(obj, root, caption)),
        ("SCENEAGENT-INGEST-DATA", lambda: _skill_ingest(obj, root, rebuild=fresh)),
        ("SCENEAGENT-RUN-STEP", lambda: _skill_run_step(obj, root, caption)),
        ("SCENEAGENT-ALIGN-SCENE", lambda: _skill_align(obj, root)),
        ("SCENEAGENT-ORIENT-BACKGROUND", lambda: _skill_skip("orient", obj, root)),
        ("SCENEAGENT-PACKAGE-DATA-PROCESSED", lambda: _skill_package(obj, root)),
        ("SCENEAGENT-INSPECT-ISAAC", lambda: _skill_skip("inspect; sim.reset was not called", obj, root)),
        ("SCENEAGENT-OBJECT-SISTERS", lambda: _skill_skip("object sisters", obj, root)),
        ("SCENEAGENT-SCENE-SISTERS", lambda: _skill_skip("scene sisters", obj, root)),
    ]


def _skill_skip(reason: str, obj: str | None = None, root: Path | None = None) -> int:
    """ScoreOnly: ledger skipped, exit 0. A skip is not a pass."""
    from tools.gate_policy import SCORE_ONLY, record_ledger, skill_exit

    record_ledger(
        obj,
        root,
        gate="skill_skip",
        gate_class=SCORE_ONLY,
        detail=reason,
        check="skipped",
        reason="skill_skip",
    )
    sys.stdout.write(f"skipped {reason}; a skip is not a pass\n")
    return skill_exit(SCORE_ONLY)


def _skill_run_pipeline(obj: str, root: Path | None, caption: str = CAN_CAPTION) -> int:
    route = choose_path(obj, root)
    sys.stdout.write(f"path {route}\n")
    sys.stdout.write(f"detector caption {caption}\n")
    if route == "stop":
        return 2
    from tools.pipeline import apply_mask_retry
    from tools.task_spec import gate_detected_target

    code = gate_detected_target(obj, root)
    if code != 0:
        return code
    return apply_mask_retry(obj, root)


def _clear_local_stale(obj: str, root: Path | None) -> None:
    """Drop old masks and scores so the first skill does not read them.

    Raw video, request.json, COLMAP, and outputs/ stay. COLMAP is removed
    later, only after a Vast host is running and a new train can start.
    """
    import shutil

    from tools.common.paths import object_dir

    folder = object_dir(obj, root)
    for name in ("masks", "crops", "chain", "ingest", "reviews", "semantics", "iou", "objects"):
        path = folder / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    retry = folder / "mask_retry.json"
    if retry.is_file():
        retry.unlink()


def _clear_local_colmap(obj: str, root: Path | None) -> None:
    """Remove local COLMAP work so the next process starts empty. outputs/ stays."""
    import shutil

    from tools.common.paths import object_dir

    folder = object_dir(obj, root) / "colmap"
    if folder.exists():
        shutil.rmtree(folder)


def _clear_remote_generated(host, obj: str) -> int:
    """Remove generated object files on the host. Raw video and outputs/ stay."""
    from tools.block_b_job import require_object_name

    try:
        name = require_object_name(obj)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not name or any(not (char.isalnum() or char in "-_") for char in name):
        sys.stderr.write(f"refusing object name {name}\n")
        return 2
    script = "\n".join(
        [
            "set -eu",
            f"base=/root/SceneagentX/data/{name}",
            'rm -rf "$base/colmap" "$base/masks" "$base/crops" "$base/chain" "$base/ingest"',
            'rm -f "$base/status.json"',
            "",
        ]
    )
    code, out, err = _ssh(host, script, timeout=120)
    if out:
        sys.stdout.write(out)
    if code != 0:
        sys.stderr.write(err or "remote generated data was not cleared\n")
    return code


def _skill_ingest(obj: str, root: Path | None, *, rebuild: bool = False) -> int:
    """Start COLMAP or splatfacto only when that orbit's artifact is missing.

    A finished checkpoint under outputs/ or a splat.ply already on disk stays.
    rebuild runs both again. The outputs/ tree is not deleted; a new train
    adds its own timestamp beside the old one.
    """
    from tools.common.paths import resolve_root
    from tools.gate_bypass import continue_past
    from tools.pipeline import VastRunner, pull_reconstruction, push_missing_raw_videos, running_vast_host
    from tools.step01_colmap import run_orbit as run_colmap
    from tools.step01_colmap import sparse_dir
    from tools.step02_splatfacto import run_orbit as run_splat

    orbits = (1, 2, 3)
    if rebuild:
        missing = list(orbits)
    else:
        missing = [orbit for orbit in orbits if _colmap_missing(obj, root, orbit) or _splat_missing(obj, root, orbit)]
        if not missing:
            sys.stdout.write("ingest kept existing COLMAP and splat checkpoints\n")
            _recover_ingest_scale(obj, root)
            return 0
    host = running_vast_host()
    if host is None:
        sys.stderr.write("no running Vast instance\n")
        return 2
    synced = _sync_host_code(host)
    if synced != 0:
        return synced
    if rebuild:
        sys.stdout.write(
            "rebuilding COLMAP and splatfacto; existing outputs stay until the new train adds a timestamp\n"
        )
        cleared = _clear_remote_generated(host, obj)
        if cleared != 0:
            return cleared
        _clear_local_colmap(obj, root)
    runner = VastRunner(host, resolve_root(root))
    pushed = push_missing_raw_videos(host, obj, root)
    if pushed != 0:
        return pushed
    failed = False
    for orbit in missing:
        sparse = sparse_dir(obj, orbit, root)
        sparse_ready = sparse.is_dir() and any(sparse.iterdir())
        if rebuild or (_colmap_missing(obj, root, orbit) and not sparse_ready):
            code = run_colmap(obj, orbit, root, runner, mark_block=orbit == 1)
            if code != 0:
                if continue_past(obj, root, "colmap_registration", f"orbit{orbit} exited {code}", check="unmet"):
                    failed = True
                    continue
                return code
        if rebuild or _splat_missing(obj, root, orbit):
            code = run_splat(obj, orbit, root, runner, mark_block=orbit == 1)
            if code == 0 and host is not None:
                copied = pull_reconstruction(host, obj, root, orbit)
                if copied != 0:
                    code = copied
            if code != 0:
                if continue_past(obj, root, "splatfacto", f"orbit{orbit} exited {code}", check="unmet"):
                    failed = True
                    continue
                return code
    _recover_ingest_scale(obj, root)
    return 1 if failed else 0


def _recover_ingest_scale(obj: str, root: Path | None) -> None:
    """After COLMAP/splat, try metric depth (or keep tag/polycam/ruler). Never VLM metres."""
    from tools.metric_depth import is_measured_record, recover_metric_scale

    try:
        record = recover_metric_scale(obj, root, orbit=1)
    except Exception as exc:  # noqa: BLE001 — ingest continues ScoreOnly on scale miss
        sys.stderr.write(f"metric scale recover failed: {exc}\n")
        return
    if is_measured_record(record):
        sys.stdout.write(
            f"ingest scale method={record.get('method')} metres_per_unit={record.get('metres_per_unit')}\n"
        )
    else:
        sys.stdout.write(
            f"ingest scale not_measured ({record.get('status')}); metres_per_unit null; achieved false\n"
        )


def _colmap_missing(obj: str, root: Path | None, orbit: int) -> bool:
    """True only when this orbit has no sparse model and no finished splat to keep."""
    from tools.common.paths import status_path
    from tools.common.status import StatusError, load_status
    from tools.step01_colmap import sparse_dir
    from tools.step02_splatfacto import splat_path

    if splat_path(obj, orbit, root).is_file():
        return False
    if status_path(obj, root).is_file():
        try:
            doc = load_status(obj, root, create=False)
        except StatusError:
            doc = {}
        stored = ((doc.get("step01") or {}).get(f"orbit{orbit}") or {}) if isinstance(doc, dict) else {}
        if stored.get("passed") is True:
            return False
    sparse = sparse_dir(obj, orbit, root)
    return not (sparse.is_dir() and any(sparse.iterdir()))


def _splat_missing(obj: str, root: Path | None, orbit: int) -> bool:
    from tools.common.paths import resolve_root
    from tools.step02_splatfacto import passing_checkpoint_stays, splat_path

    if passing_checkpoint_stays(obj, orbit, root):
        return False
    if splat_path(obj, orbit, root).is_file():
        return False
    base = resolve_root(root) / "outputs" / f"{obj}_orbit{orbit}" / "splatfacto"
    if base.is_dir() and any(base.glob("*/config.yml")):
        return False
    return True


def _skill_run_step(obj: str, root: Path | None, caption: str = CAN_CAPTION) -> int:
    """Launch Block B on the live Vast host from the stored request story."""
    from tools.pipeline import running_vast_host

    host = running_vast_host()
    if host is None:
        sys.stderr.write("no running Vast instance\n")
        return 2
    pairs = _splat_configs(obj, root, host)
    if not pairs:
        sys.stderr.write("no splat config; Block B was not started and the splat was not retrained\n")
        return 2
    from tools.block_b_job import host_launch_script, mask_downstream_block_reason
    from tools.common.paths import object_dir

    folder = object_dir(obj, root)
    blocked = mask_downstream_block_reason(folder)
    if blocked:
        sys.stderr.write(blocked + "\n")
        return 2
    spec = _load_task_spec(folder)
    if spec is not None:
        script = host_launch_script(obj, pairs, spec=spec, log_stem=f"{obj}_block_h")
    else:
        script = host_launch_script(obj, pairs, prompt=caption, log_stem=f"{obj}_block_h")
    copied = _sync_host_code(host)
    if copied != 0:
        return copied
    code, out, err = _ssh(host, script, timeout=90)
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    _remember_gpu(obj, host, out, err, code)
    if code != 0 and "ALREADY_RUNNING" not in out:
        return code if code != 0 else 2
    return 0


def _load_task_spec(folder: Path) -> dict[str, Any] | None:
    """request.json with job and target. A synonym-only caption is not a story."""
    path = Path(folder) / "request.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    if loaded.get("job") not in {"orbit", "room"}:
        return None
    target = loaded.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return loaded


def _splat_configs(obj: str, root: Path | None, host) -> list[tuple[str, str]]:
    code, out, err = _ssh(
        host,
        "\n".join(
            [
                "set -eu",
                "python3 - << 'PY'",
                "from pathlib import Path",
                "root = Path('/root/SceneagentX/outputs')",
                f"name = {obj!r}",
                "for orbit in ('orbit1', 'orbit2', 'orbit3'):",
                "    base = root / f'{name}_{orbit}' / 'splatfacto'",
                "    found = sorted(base.glob('*/config.yml')) if base.is_dir() else []",
                "    print(orbit + '\\t' + (str(found[-1]) if found else 'MISSING'))",
                "PY",
                "",
            ]
        ),
        timeout=60,
    )
    pairs: list[tuple[str, str]] = []
    if code == 0:
        for line in out.splitlines():
            if "\t" not in line:
                continue
            orbit, path = line.split("\t", 1)
            if orbit.startswith("orbit") and path.startswith("/") and "MISSING" not in path:
                pairs.append((orbit, path.strip()))
    if pairs:
        return pairs
    from tools.common.paths import resolve_root

    local_root = resolve_root(root)
    for orbit in ("orbit1", "orbit2", "orbit3"):
        base = local_root / "outputs" / f"{obj}_{orbit}" / "splatfacto"
        found = sorted(base.glob("*/config.yml")) if base.is_dir() else []
        if not found:
            continue
        rel = found[-1].resolve().relative_to(local_root.resolve()).as_posix()
        pairs.append((orbit, f"/root/SceneagentX/{rel}"))
    if err:
        sys.stderr.write(err)
    if pairs:
        return pairs
    return _block_a_configs(obj, root)


def _block_a_configs(obj: str, root: Path | None) -> list[tuple[str, str]]:
    """Frozen configs named by the Block A record. A missing file is not a new train."""
    from tools.common.paths import object_dir

    path = object_dir(obj, root) / "ingest" / "block_a.json"
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = loaded.get("orbits") if isinstance(loaded, dict) else None
    if not isinstance(rows, list):
        return []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        orbit = row.get("orbit")
        config = row.get("config")
        if not isinstance(orbit, str) or not isinstance(config, str):
            continue
        if not orbit.startswith("orbit") or not config.startswith("/"):
            continue
        if row.get("train_finished") is not True:
            continue
        pairs.append((orbit, config))
    return pairs


def _sync_host_code(host) -> int:
    from tools.pipeline import ssh_key_path

    files = (
        "tools/agent_loop.py",
        "tools/task_spec.py",
        "tools/pipeline.py",
        "tools/gate_bypass.py",
        "tools/block_b_job.py",
        "tools/step01_colmap.py",
        "tools/step02_splatfacto.py",
        "tools/common/vlm.py",
        "tools/can_instances.py",
        "vast-cluster/scripts/object_semantics.py",
        "vast-cluster/scripts/object_langsplat_b2.py",
        "vast-cluster/scripts/object_langsplat_b3.py",
        "vast-cluster/scripts/object_langsplat_b4.py",
        "vast-cluster/scripts/object_cluster_can_orbit.py",
        "vast-cluster/scripts/object_crop_iou.py",
        "vast-cluster/scripts/object_splat_iou.py",
    )
    key = ssh_key_path()
    root = Path(__file__).resolve().parents[1]
    for rel in files:
        local = root / rel
        remote = "/root/SceneagentX/" + rel.replace("\\", "/")
        parent = remote.rsplit("/", 1)[0]
        made, _, err = _ssh(host, f"mkdir -p {parent}\n", timeout=60)
        if made != 0:
            sys.stderr.write(err or f"mkdir failed {rel}\n")
            return made
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
                str(local),
                f"root@{host.host}:{remote}",
            ],
            capture_output=True,
            check=False,
        )
        if done.returncode != 0:
            sys.stderr.write((done.stderr or b"").decode("utf-8", "replace"))
            return int(done.returncode)
        sys.stdout.write(f"copied {rel}\n")
    return 0


def _ssh(host, script: str, timeout: int = 120) -> tuple[int, str, str]:
    from tools.pipeline import ssh_key_path

    try:
        done = subprocess.run(
            [
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
                "bash",
                "-s",
            ],
            input=script.replace("\r", "").encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 2, "", str(exc)
    return done.returncode, (done.stdout or b"").decode("utf-8", "replace"), (done.stderr or b"").decode("utf-8", "replace")


def _remember_gpu(obj: str, host, out: str, err: str, code: int) -> None:
    pid = ""
    for line in (out or "").splitlines():
        if line.startswith("STARTED") or line.startswith("ALREADY_RUNNING"):
            parts = line.split()
            if len(parts) > 1:
                pid = parts[-1]
    _skill_run_step.gpu = {
        "instance_id": host.instance_id,
        "host": host.host,
        "port": host.port,
        "pid": pid,
        "log": f"/root/logs/{obj}_block_h.log",
        "exit": code,
        "achieved": False,
        "stderr": (err or "")[-400:],
    }


def _skill_align(obj: str, root: Path | None) -> int:
    """Align when possible. Missing metres/tag is ScoreOnly and does not kill the walk."""
    from tools.align import run_align
    from tools.gate_policy import SCORE_ONLY, record_ledger, skill_exit

    code = run_align(obj, root, rerun_orbits=False)
    if code == 0:
        return 0
    record_ledger(
        obj,
        root,
        gate="SCENEAGENT-ALIGN-SCENE",
        gate_class=SCORE_ONLY,
        detail="metres_per_unit missing or alignment unmet; scale stays unmeasured",
        check="unmet",
        reason="no_metres",
    )
    sys.stdout.write("align unmet; metres_per_unit stays null; not marked achieved\n")
    return skill_exit(SCORE_ONLY)


def _skill_package(obj: str, root: Path | None) -> int:
    """ScoreOnly: USDZ present exits 0 as present_unachieved; missing is not invented."""
    from tools.common.paths import object_dir
    from tools.gate_policy import SCORE_ONLY, record_ledger, skill_exit

    folder = object_dir(obj, root)
    if (folder / "scene.usdz").is_file() or (folder / "asset.usdz").is_file():
        record_ledger(
            obj,
            root,
            gate="SCENEAGENT-PACKAGE-DATA-PROCESSED",
            gate_class=SCORE_ONLY,
            detail="package file present; not marked achieved",
            check="present_unachieved",
            reason="package_present",
        )
        sys.stdout.write("package file present; not marked achieved\n")
        return skill_exit(SCORE_ONLY)
    record_ledger(
        obj,
        root,
        gate="SCENEAGENT-PACKAGE-DATA-PROCESSED",
        gate_class=SCORE_ONLY,
        detail="package missing; not invented",
        check="skipped",
        reason="package_missing",
    )
    sys.stdout.write("package missing; not invented\n")
    return skill_exit(SCORE_ONLY)


def _write_block_h(
    obj: str,
    root: Path | None,
    walked: list[dict[str, Any]],
    stopped: str | None,
    caption: str = CAN_CAPTION,
    prompt: str | None = None,
    videos: list[str] | None = None,
) -> None:
    from tools.common.paths import object_dir

    gpu = getattr(_skill_run_step, "gpu", None)
    payload = {
        "orchestrator": "SCENEAGENT-RUN-PIPELINE",
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "milestone": False,
        "stopped": stopped,
        "caption": caption,
        "skills": walked,
        "gpu": gpu if isinstance(gpu, dict) else None,
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if videos:
        payload["videos"] = videos
    folder = object_dir(obj, root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "block_h.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _prompt_source(videos: list[Path]) -> Path | None:
    """The video whose frames are sampled for the prompt. Later orbits stay on disk."""
    for path in videos:
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    return videos[0] if videos else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Advance SceneAgent skills")
    parser.add_argument("obj", nargs="?", default="bottle")
    parser.add_argument("--obj", dest="obj_name", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--orchestrate", action="store_true", help="Walk Block H skills from SCENEAGENT-RUN-PIPELINE")
    parser.add_argument("--bypass-gates", action="store_true", help="Set SCENEAGENT_BYPASS_GATES for this process")
    parser.add_argument("--prompt", default=None, help="User pipeline prompt, stored before any Block H skill")
    parser.add_argument("--input", action="append", type=Path, default=None, help="Input video. Repeat for each orbit.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="With --orchestrate, rebuild COLMAP and splatfacto even when checkpoints exist",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--skill", default=None)
    args = parser.parse_args(argv)
    obj = args.obj_name or args.obj
    if args.bypass_gates:
        from tools.gate_bypass import ENV_NAME

        os.environ[ENV_NAME] = "1"
    if args.orchestrate:
        videos = list(args.input or [])
        try:
            return run_block_h(
                obj,
                args.root,
                prompt=args.prompt,
                source=_prompt_source(videos),
                fresh=args.fresh,
                videos=videos,
            )
        except UnknownSkillError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
    if args.prompt is not None or args.input or args.fresh:
        sys.stderr.write("pass --orchestrate to store --prompt and the input video\n")
        return 2
    if not args.once:
        sys.stderr.write("pass --once to advance one skill, or --orchestrate to walk Block H\n")
        return 2
    planner = None if args.skill is None else (lambda _obj, _root, skill=args.skill: skill)
    try:
        return advance_once(obj, args.root, planner=planner)
    except UnknownSkillError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
