"""Resume one skill, and repair it only when the retry changes something.

Measurements judge registration, reprojection, scale, and surface error.
A vision model is asked only for what eyes have to judge. A retry that
repeats the last try is refused. A worse measurement is thrown away.
Two tries is the limit. The next step is one question, not a third try.
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
import urllib.error
import urllib.request
from typing import Any, Callable

from tools.common.paths import object_dir, renders_dir
from tools.harvard_agents import write_agents
from tools.ingest import classify_input, colmap_refused
from tools.review_step import REQUIRED, ReviewError, review_block
from tools.video_input import find_video, record_absent

MAX_FAULTS = 2
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
        restarted = faults < MAX_FAULTS
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
                f"{block_id} used its two tries. What should change on the next try?",
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

    load_status(obj, root, create=True)
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
