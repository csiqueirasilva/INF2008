from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Canonical colour map used across overlay helpers (BGR tuples).
_BASE_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 0, 0),
    2: (255, 128, 0),
    3: (255, 255, 0),
    4: (0, 255, 0),
    5: (0, 255, 255),
    6: (0, 128, 255),
    7: (0, 0, 255),
}


def label_to_color(lid: int) -> Tuple[int, int, int]:
    """Return a BGR colour for *lid*. Falls back to a deterministic colour."""
    lid = int(lid)
    if lid in _BASE_COLORS:
        return _BASE_COLORS[lid]
    rng = np.random.default_rng(lid)
    rgb = rng.integers(32, 224, size=3, endpoint=True)
    # numpy gives RGB; OpenCV expects BGR
    return int(rgb[2]), int(rgb[1]), int(rgb[0])
