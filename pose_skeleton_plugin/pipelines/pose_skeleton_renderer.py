"""Pure rendering + smoothing helpers for pose skeleton images.

This module is intentionally deterministic and testable without MediaPipe/OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PoseLandmark:
    """Pose landmark in normalized coordinates (0..1)."""

    x: float
    y: float
    score: float = 1.0


class LandmarkSmoother:
    """Simple EMA smoother over landmark positions."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._state: list[PoseLandmark] | None = None

    def update(self, landmarks: Sequence[PoseLandmark] | None) -> list[PoseLandmark] | None:
        if landmarks is None:
            self._state = None
            return None

        if self._state is None or len(self._state) != len(landmarks) or self.alpha <= 0.0:
            self._state = list(landmarks)
            return list(landmarks)

        a = self.alpha
        out: list[PoseLandmark] = []
        for prev, cur in zip(self._state, landmarks, strict=False):
            out.append(
                PoseLandmark(
                    x=(1.0 - a) * cur.x + a * prev.x,
                    y=(1.0 - a) * cur.y + a * prev.y,
                    score=cur.score,
                )
            )
        self._state = out
        return list(out)


def render_skeleton(
    *,
    height: int,
    width: int,
    landmarks: Sequence[PoseLandmark] | None,
    connections: Iterable[tuple[int, int]],
    min_confidence: float,
    thickness: int,
    joint_radius: int,
) -> np.ndarray:
    """Render a skeleton image (H, W, 3) float32 in [0, 1], black background."""

    img = np.zeros((int(height), int(width), 3), dtype=np.float32)
    if not landmarks:
        return img

    min_conf = float(min_confidence)
    thick = max(1, int(thickness))
    jr = max(0, int(joint_radius))

    # Convert to pixel coords once.
    pts: list[tuple[int, int] | None] = []
    scores: list[float] = []
    for lm in landmarks:
        scores.append(float(lm.score))
        if float(lm.score) < min_conf:
            pts.append(None)
            continue
        x = int(round(float(lm.x) * (width - 1)))
        y = int(round(float(lm.y) * (height - 1)))
        pts.append((x, y))

    # Draw connections (white).
    for a, b in connections:
        if a < 0 or b < 0 or a >= len(pts) or b >= len(pts):
            continue
        pa = pts[a]
        pb = pts[b]
        if pa is None or pb is None:
            continue
        _draw_thick_line(img, pa[0], pa[1], pb[0], pb[1], radius=max(0, thick // 2))

    # Draw joints.
    if jr > 0:
        for p in pts:
            if p is None:
                continue
            _draw_disk(img, p[0], p[1], radius=jr)

    return img


def _draw_disk(img: np.ndarray, cx: int, cy: int, radius: int) -> None:
    if radius <= 0:
        if 0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]:
            img[cy, cx, :] = 1.0
        return

    H, W = img.shape[0], img.shape[1]
    r = int(radius)
    x0 = max(0, cx - r)
    x1 = min(W - 1, cx + r)
    y0 = max(0, cy - r)
    y1 = min(H - 1, cy + r)
    if x1 < x0 or y1 < y0:
        return

    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[y0 : y1 + 1, x0 : x1 + 1][mask] = 1.0


def _draw_thick_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int) -> None:
    # Sample points along the line and stamp disks. Simple + deterministic.
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy), 1))
    xs = np.linspace(x0, x1, num=steps + 1)
    ys = np.linspace(y0, y1, num=steps + 1)
    for x, y in zip(xs, ys, strict=False):
        _draw_disk(img, int(round(x)), int(round(y)), radius=radius)

