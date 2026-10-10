"""Name bottle parts from one vision reply.

PartField chooses a count only through tools.partfield_parts.choose_k.
A missing or losing score stays one body. The vision reply may name parts
from pictures. A second call happens only when that reply asks for a new count.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_MODEL = "gpt-6-astra"
_URL = "https://api.openai.com/v1/chat/completions"


def parse_part_reply(payload: dict[str, Any], current_count: int) -> dict[str, Any]:
    """Keep part names and a requested count only when it is a real change."""
    names: list[str] = []
    seen: set[str] = set()
    for item in payload.get("parts") or []:
        if isinstance(item, str):
            raw = item
            note = ""
        elif isinstance(item, dict):
            raw = str(item.get("id") or item.get("name") or "")
            note = str(item.get("note") or "")[:160]
        else:
            continue
        name = _slug(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name if not note else name)
    requested = payload.get("requested_part_count")
    change: int | None = None
    if isinstance(requested, int) and not isinstance(requested, bool) and requested >= 1 and requested != current_count:
        change = requested
    elif isinstance(requested, float) and requested.is_integer() and int(requested) >= 1 and int(requested) != current_count:
        change = int(requested)
    return {
        "parts": names,
        "split_wrong": payload.get("split_wrong") is True,
        "requested_part_count": change,
        "object": str(payload.get("object") or "bottle")[:80],
    }


def json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("vision JSON must be an object")
    return parsed


def ask_parts(images: list[Path], current_count: int, note: str) -> dict[str, Any]:
    """One gpt-6-astra look. No key means the model was not asked."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or not images:
        return {"called": False, "parts": [], "split_wrong": False, "requested_part_count": None, "object": "bottle"}
    text = (
        "These are colored views of one bottle. "
        f"The coarse split currently has {current_count} part(s). {note} "
        "Name the visible parts. Reply with JSON only: "
        '{"object":"bottle","parts":[{"id":"cap","note":"short"}],'
        '"split_wrong":false,"requested_part_count":null} '
        "Set requested_part_count to an integer only when the part count should change. "
        "Do not invent a measured joint, a scale weight, or a second capture."
    )
    try:
        try:
            body = _post(key, images, text, json_mode=True)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            body = _post(key, images, text, json_mode=False)
        parsed = parse_part_reply(json_object(_message_text(body)), current_count)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {"called": False, "parts": [], "split_wrong": False, "requested_part_count": None, "object": "bottle"}
    parsed["called"] = True
    return parsed


def _post(api_key: str, images: list[Path], text: str, json_mode: bool) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images[:4]:
        content.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
    payload: dict[str, Any] = {"model": _MODEL, "messages": [{"role": "user", "content": content}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        _URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _data_url(path: Path) -> str:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"unreadable image {path.name}")
    height, width = image.shape[:2]
    if width > 640:
        scale = 640 / width
        image = cv2.resize(image, (640, int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise OSError(f"could not encode {path.name}")
    return "data:image/jpeg;base64," + base64.standard_b64encode(encoded.tobytes()).decode("ascii")


def _message_text(body: dict[str, Any]) -> str:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("vision message content is missing")
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        return ""
    return cleaned[:40]


def place_pivots(parts: list[dict[str, Any]], points: Any) -> None:
    """Put a pivot on the long axis only after a joint type is already named."""
    import numpy as np

    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) < 10:
        return
    low = np.percentile(cloud, 5, axis=0)
    high = np.percentile(cloud, 95, axis=0)
    axis = int(np.argmax(high - low))
    center = np.median(cloud, axis=0)
    for part in parts:
        if part.get("joint") not in {"prismatic", "revolute"}:
            continue
        origin = center.copy()
        origin[axis] = high[axis] if part["joint"] == "revolute" else center[axis]
        part["origin"] = [round(float(value), 6) for value in origin]


def name_parts(images: list[Path]) -> dict[str, Any]:
    """Coarse one-body split, one vision look, and one refinement if the count changes."""
    from tools.partfield_parts import choose_k

    chosen = choose_k(None)
    current = 1 if chosen is None else chosen
    note = "No PartField score beat k=1, so the coarse split was not given extra parts."
    first = ask_parts(images, current, note)
    if not first.get("called"):
        raise RuntimeError("gpt-6-astra was not asked or did not return parts")
    final = first
    refined = False
    if first.get("requested_part_count"):
        count = int(first["requested_part_count"])
        final = ask_parts(images, count, f"Name exactly {count} parts. This is the only refinement.")
        if not final.get("called"):
            raise RuntimeError("the part-count refinement did not return")
        refined = True
    return {"coarse_k": current, "partfield_k": chosen, "refined": refined, "reply": final}
