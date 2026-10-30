"""Print a Vast create command for the SceneAgent GPU box.

Does not rent an instance and does not read or write an API key.
vastai, if you run a live search, uses the key already stored by `vastai set api-key`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.vast_search import main

if __name__ == "__main__":
    raise SystemExit(main())
