"""Repo-root CHANGELOG.md. Closed block sections are appended around, never rewritten."""

from __future__ import annotations

import re
from pathlib import Path

_UNRELEASED = "## [Unreleased]"
_BLOCK = re.compile(r"^## \[block-\d+]", re.MULTILINE)
_KINDS = ("Added", "Changed", "Fixed")


class ChangelogError(RuntimeError):
    """The changelog rule refused the edit."""


def append_unreleased(path: Path, kind: str, bullet: str) -> None:
    """Add one bullet under [Unreleased]. Closed [block-NN] sections stay byte-for-byte."""
    heading = kind.strip().capitalize()
    if heading not in _KINDS:
        raise ChangelogError("changelog bullets go under Added, Changed, or Fixed")
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Changelog\n\n## [Unreleased]\n"
    closed = _closed_sections(text)
    if _UNRELEASED not in text:
        text = text.rstrip() + "\n\n## [Unreleased]\n"
    line = bullet if bullet.startswith("- ") else f"- {bullet}"
    text = _insert_bullet(text, heading, line)
    if _closed_sections(text) != closed:
        raise ChangelogError("refusing to rewrite a closed changelog section")
    path.write_text(text, encoding="utf-8")


def close_block(
    path: Path,
    block_id: str,
    date: str,
    title: str,
    milestone: list[str],
    tests: list[str],
    *,
    artifact_present: bool,
) -> None:
    """Move [Unreleased] into a block section. Unreleased is empty afterwards."""
    if not milestone or not tests:
        raise ChangelogError("a block section needs Milestone and Tests")
    _validate_milestone(milestone, tests, artifact_present)
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Changelog\n\n## [Unreleased]\n"
    if _UNRELEASED not in text:
        raise ChangelogError("changelog is missing [Unreleased]")
    start = text.find(_UNRELEASED)
    next_heading = text.find("\n## [", start + len(_UNRELEASED))
    end = len(text) if next_heading < 0 else next_heading
    body = text[start + len(_UNRELEASED) : end].strip("\n")
    if not body.strip():
        raise ChangelogError("[Unreleased] is empty; there is nothing to close")
    section = (
        f"## [{block_id}] {date} — {title}\n"
        + body.strip()
        + "\n### Milestone\n"
        + "\n".join(line if line.startswith("- ") else f"- {line}" for line in milestone)
        + "\n### Tests\n"
        + "\n".join(line if line.startswith("- ") else f"- {line}" for line in tests)
        + "\n"
    )
    rebuilt = text[:start] + _UNRELEASED + "\n\n" + section + text[end:].lstrip("\n")
    if not rebuilt[rebuilt.find(_UNRELEASED) + len(_UNRELEASED) :].lstrip().startswith("## ["):
        # Unreleased must have no leftover bullets.
        unreleased_body = _unreleased_body(rebuilt)
        if unreleased_body.strip():
            raise ChangelogError("[Unreleased] is non-empty at a block close")
    path.write_text(rebuilt if rebuilt.endswith("\n") else rebuilt + "\n", encoding="utf-8")
    if _unreleased_body(path.read_text(encoding="utf-8")).strip():
        raise ChangelogError("[Unreleased] is non-empty at a block close")


def validate_changelog(text: str, artifact_present: bool) -> None:
    """Fail when a closed block lacks Milestone or Tests, or claims achieved without a file."""
    for heading, body in _sections(text):
        if not heading.startswith("block-"):
            continue
        if "### Milestone" not in body or "### Tests" not in body:
            raise ChangelogError(f"{heading} lacks Milestone or Tests")
        milestone = _subsection(body, "Milestone")
        tests = _subsection(body, "Tests")
        _validate_milestone(milestone.splitlines(), tests.splitlines(), artifact_present)


def _validate_milestone(milestone: list[str], tests: list[str], artifact_present: bool) -> None:
    blob = "\n".join(milestone).lower()
    tests_blob = "\n".join(tests).lower()
    if "achieved" in blob and ("skipped" in blob or "unstarted" in blob or "asked: false" in blob or "asked:false" in blob):
        raise ChangelogError("skipped, unstarted, and asked: false are not achieved")
    if "achieved" in blob and not artifact_present:
        raise ChangelogError("milestone says achieved but the artifact file is missing")
    if "achieved" in blob and "passed" not in tests_blob:
        raise ChangelogError("an achieved milestone needs a passed test in Tests")


def _insert_bullet(text: str, heading: str, line: str) -> str:
    start = text.find(_UNRELEASED)
    next_heading = text.find("\n## [", start + len(_UNRELEASED))
    end = len(text) if next_heading < 0 else next_heading
    head = text[:start]
    body = text[start:end]
    tail = text[end:]
    marker = f"### {heading}\n"
    if marker not in body:
        body = body.rstrip() + f"\n\n### {heading}\n{line}\n"
    else:
        at = body.find(marker) + len(marker)
        body = body[:at] + line + "\n" + body[at:]
    if not body.endswith("\n"):
        body += "\n"
    return head + body + tail


def _closed_sections(text: str) -> list[str]:
    return [body for heading, body in _sections(text) if heading.startswith("block-")]


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^## \[([^\]]+)]", text, re.MULTILINE))
    sections = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1), text[start:end]))
    return sections


def _unreleased_body(text: str) -> str:
    for heading, body in _sections(text):
        if heading == "Unreleased":
            return body
    return ""


def _subsection(body: str, name: str) -> str:
    marker = f"### {name}"
    start = body.find(marker)
    if start < 0:
        return ""
    rest = body[start + len(marker) :]
    nxt = rest.find("\n### ")
    return rest if nxt < 0 else rest[:nxt]
