"""Choose a PartField part count only when a score supports the split."""

from __future__ import annotations


def choose_k(scores: dict[int, float] | None) -> int | None:
    """Return a part count greater than one, or None to keep a single body.

    ``scores`` maps a candidate k to a higher-is-better score. A split is
    supported only when k=1 is in the sweep and some k greater than 1 scores
    strictly above it. A sweep that never scores the single body does not
    choose a k.
    """
    if not scores:
        return None
    usable = {int(k): float(value) for k, value in scores.items() if int(k) >= 1}
    if 1 not in usable:
        return None
    best = max(usable, key=lambda key: (usable[key], -key))
    if best <= 1 or usable[best] <= usable[1]:
        return None
    return best
