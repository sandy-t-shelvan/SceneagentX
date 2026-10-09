"""Read and write data/<object>/status.json.

The file is a nested object so the plan's jq paths work
(.step01.orbit1.registered_pct, .ground_truth.length_m, .setup_minutes).
`steps` is a list of {id, name, state, attempts} for the B16 agent loop.
The plan's `jq '[.[] | .state]'` assumes an array root, which cannot also
serve those nested paths. Use `jq '[.steps[] | .state]'`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from tools.common.blocks import BLOCKS, get_block
from tools.common.paths import schema_path, status_path

STATES = ("not_started", "running", "pass", "fail")


class StatusError(Exception):
    """The status file is missing a value or fails the schema."""


def default_status(obj: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "object": obj,
        "ground_truth": {"length_m": None},
        "setup_minutes": None,
        "steps": [
            {
                "id": block.id,
                "name": block.name,
                "state": "not_started",
                "attempts": 0,
            }
            for block in BLOCKS
        ],
    }


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def validate_status(doc: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    validator = Draft202012Validator(schema or load_schema())
    errors = sorted(validator.iter_errors(doc), key=lambda err: list(err.path))
    if not errors:
        length = (doc.get("ground_truth") or {}).get("length_m")
        if length is not None and length <= 0:
            raise StatusError("ground_truth.length_m must be > 0 metres when set")
        return
    raise StatusError(_format_errors(errors))


def _format_errors(errors: list[ValidationError]) -> str:
    lines = []
    for err in errors:
        where = ".".join(str(part) for part in err.path) or "(root)"
        lines.append(f"{where}: {err.message}")
    return "status.json failed schema validation:\n" + "\n".join(lines)


def _merge_steps(saved: list[Any]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in saved:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            by_id[item["id"]] = item
    merged: list[dict[str, Any]] = []
    known: set[str] = set()
    for block in BLOCKS:
        base = {
            "id": block.id,
            "name": block.name,
            "state": "not_started",
            "attempts": 0,
        }
        if block.id in by_id:
            base.update(by_id[block.id])
            base["id"] = block.id
            base["name"] = by_id[block.id].get("name") or block.name
        merged.append(base)
        known.add(block.id)
    for item in saved:
        step_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(step_id, str) and step_id not in known:
            merged.append(item)
    return merged


def merge_defaults(saved: dict[str, Any], obj: str) -> dict[str, Any]:
    doc = default_status(obj)
    for key, value in saved.items():
        if key in ("steps", "ground_truth"):
            continue
        doc[key] = value
    doc["object"] = saved.get("object") or obj
    ground = dict(doc["ground_truth"])
    ground.update(saved.get("ground_truth") or {})
    doc["ground_truth"] = ground
    doc["steps"] = _merge_steps(list(saved.get("steps") or []))
    return doc


def load_status(obj: str, root: Path | None = None, create: bool = True) -> dict[str, Any]:
    path = status_path(obj, root)
    if not path.exists():
        if not create:
            raise StatusError(f"No status file at {path}")
        doc = default_status(obj)
        save_status(obj, doc, root)
        return doc
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise StatusError(f"{path} must be a JSON object")
    doc = merge_defaults(saved, obj)
    validate_status(doc)
    return doc


def save_status(obj: str, doc: dict[str, Any], root: Path | None = None) -> Path:
    validate_status(doc)
    path = status_path(obj, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2, sort_keys=False) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


def find_step(doc: dict[str, Any], block_id: str) -> dict[str, Any]:
    block = get_block(block_id)
    for step in doc["steps"]:
        if step["id"] == block.id:
            return step
    raise StatusError(f"Block {block.id} is missing from status.json")


def set_path(doc: dict[str, Any], dotted: str, value: Any) -> None:
    if not dotted:
        raise StatusError("A status path is required")
    parts = dotted.split(".")
    cursor = doc
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def get_path(doc: dict[str, Any], dotted: str) -> Any:
    cursor: Any = doc
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise StatusError(f"status.json has no value at {dotted}")
        cursor = cursor[part]
    return cursor


def set_metric(obj: str, dotted: str, value: Any, root: Path | None = None) -> dict[str, Any]:
    doc = load_status(obj, root)
    set_path(doc, dotted, value)
    save_status(obj, doc, root)
    return doc


def begin_step(obj: str, block_id: str, root: Path | None = None) -> dict[str, Any]:
    doc = load_status(obj, root)
    step = find_step(doc, block_id)
    step["attempts"] = int(step.get("attempts") or 0) + 1
    step["state"] = "running"
    save_status(obj, doc, root)
    return doc


def finish_step(
    obj: str,
    block_id: str,
    state: str,
    root: Path | None = None,
    actual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in ("pass", "fail"):
        raise StatusError(f"finish state must be pass or fail, got {state!r}")
    doc = load_status(obj, root)
    step = find_step(doc, block_id)
    if step.get("state") != "running":
        step["attempts"] = int(step.get("attempts") or 0) + 1
    step["state"] = state
    if actual is not None:
        step["actual"] = actual
    save_status(obj, doc, root)
    return doc


def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _print_json(doc: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(doc, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read and update data/<object>/status.json")
    parser.add_argument("--obj", default="marker", help="Object name under data/")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to this checkout.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create status.json if it does not exist")
    sub.add_parser("show", help="Print status.json")

    set_cmd = sub.add_parser("set", help="Set a dotted path, e.g. ground_truth.length_m")
    set_cmd.add_argument("path")
    set_cmd.add_argument("value", help="JSON value (number, string, object) or plain text")

    minutes = sub.add_parser("set-setup-minutes", help="Record how long setup.sh took")
    minutes.add_argument("--minutes", type=float, required=True)

    begin = sub.add_parser("begin", help="Mark a block running and increment attempts")
    begin.add_argument("--step", required=True)

    finish = sub.add_parser("finish", help="Mark a block pass or fail")
    finish.add_argument("--step", required=True)
    finish.add_argument("--state", required=True, choices=("pass", "fail"))

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            doc = load_status(args.obj, args.root, create=True)
        elif args.command == "show":
            doc = load_status(args.obj, args.root, create=True)
        elif args.command == "set":
            doc = set_metric(args.obj, args.path, parse_value(args.value), args.root)
        elif args.command == "set-setup-minutes":
            if args.minutes < 0:
                raise StatusError("setup minutes cannot be negative")
            doc = set_metric(args.obj, "setup_minutes", args.minutes, args.root)
        elif args.command == "begin":
            doc = begin_step(args.obj, args.step, args.root)
        else:
            doc = finish_step(args.obj, args.step, args.state, args.root)
    except (StatusError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    _print_json(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
