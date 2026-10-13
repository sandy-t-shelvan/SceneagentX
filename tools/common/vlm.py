"""Laptop vision gate for one SceneAgent step.

The OpenAI key stays in this process. This module does not SSH, does not
copy the key onto a Vast host, and does not write the key into the review file.
A missing picture raises MissingReviewImage so the caller can record asked false.
That path is never a pass.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL = "gpt-6-astra"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "confidence", "problem", "correction", "evidence"],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "problem": {
            "type": "string",
            "enum": [
                "none",
                "scale",
                "rotation",
                "placement",
                "missing_part",
                "extra_part",
                "wrong_label",
                "seam",
                "floater",
                "penetration",
                "other",
            ],
        },
        "correction": {
            "type": "object",
            "additionalProperties": False,
            "required": ["parameter", "value", "reason"],
            "properties": {
                "parameter": {"type": "string"},
                "value": {"type": ["number", "string", "null"]},
                "reason": {"type": "string"},
            },
        },
        "evidence": {"type": "string"},
    },
}

_DISALLOWED = "model proposed a disallowed parameter"


class VlmError(RuntimeError):
    """The vision call could not be trusted. This is not a pass."""


class MissingReviewImage(VlmError):
    """The review picture is not on disk. The gate records asked: false."""


def review(
    step_id: str,
    images: list[str | Path],
    question: str,
    allowed_params: list[str] | tuple[str, ...],
    record_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ask for one verdict. temperature is 0. A low-confidence pass becomes a fail."""
    paths = [Path(image) for image in images]
    if not paths:
        raise MissingReviewImage("missing review image: no picture was provided")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise MissingReviewImage(f"missing review image: {missing[0]}")

    allowed = [str(name) for name in allowed_params]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": question + "\nAllowed parameters: " + ", ".join(allowed),
        }
    ]
    for path in paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

    model = os.environ.get("SCENEAGENT_VLM_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA},
        },
    }
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise VlmError("OPENAI_API_KEY is not set on the laptop")

    body = _http_post(_URL, payload, api_key, timeout=120)
    try:
        raw = body["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VlmError("vision response was not the verdict schema") from exc
    if not isinstance(parsed, dict):
        raise VlmError("vision response was not the verdict schema")

    verdict = _normalize(parsed, allowed)
    if record_path is not None:
        _write_record(Path(record_path), step_id, payload, body, verdict)
    return verdict


def _normalize(parsed: dict[str, Any], allowed: list[str]) -> dict[str, Any]:
    verdict = str(parsed.get("verdict") or "")
    if verdict not in {"pass", "fail"}:
        raise VlmError("vision verdict must be pass or fail")
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise VlmError("vision confidence must be a number") from exc
    problem = str(parsed.get("problem") or "other")
    correction = parsed.get("correction") if isinstance(parsed.get("correction"), dict) else {}
    parameter = correction.get("parameter")
    value = correction.get("value")
    reason = str(correction.get("reason") or "")
    evidence = str(parsed.get("evidence") or "")

    if verdict == "pass" and confidence < 0.6:
        verdict = "fail"
        problem = "other"

    if verdict == "fail" and parameter not in allowed:
        parameter = None
        value = None
        reason = _DISALLOWED

    return {
        "verdict": verdict,
        "confidence": confidence,
        "problem": problem,
        "correction": {"parameter": parameter, "value": value, "reason": reason},
        "evidence": evidence,
    }


def _write_record(
    path: Path,
    step_id: str,
    payload: dict[str, Any],
    body: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step_id,
        "request": payload,
        "response": body,
        "verdict": verdict,
    }
    text = json.dumps(record, indent=2) + "\n"
    if os.environ.get("OPENAI_API_KEY", "").strip() and os.environ["OPENAI_API_KEY"].strip() in text:
        raise VlmError("review file refused to store the API key")
    path.write_text(text, encoding="utf-8")


def _http_post(url: str, payload: dict[str, Any], api_key: str, timeout: int = 120) -> dict[str, Any]:
    """POST chat completions. Tests replace this. The key is not written to disk."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if (
            exc.code == 400
            and "temperature" in body
            and "temperature" in payload
        ):
            retry = dict(payload)
            retry.pop("temperature", None)
            return _http_post(url, retry, api_key, timeout)
        raise VlmError(f"vision HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise VlmError("vision request failed") from exc
    if not isinstance(parsed, dict):
        raise VlmError("vision response was not an object")
    return parsed
