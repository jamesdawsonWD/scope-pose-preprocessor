"""MediaPipe pose estimator wrapper.

Kept thin so the renderer + unit tests remain deterministic and dependency-light.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .pose_skeleton_renderer import PoseLandmark


def get_mediapipe_pose_connections() -> list[tuple[int, int]]:
    """Return MediaPipe POSE_CONNECTIONS as pairs of landmark indices."""
    import mediapipe as mp

    conns: Iterable[tuple[object, object]] = mp.solutions.pose.POSE_CONNECTIONS
    out: list[tuple[int, int]] = []
    for a, b in conns:
        # MediaPipe uses an enum-like class; int() yields the index.
        out.append((int(a), int(b)))
    return out


class MediaPipePoseEstimator:
    """Estimate a single person's pose landmarks from an RGB frame."""

    def __init__(self):
        import mediapipe as mp

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame_rgb_u8: np.ndarray) -> list[PoseLandmark] | None:
        """Return landmarks in normalized coords, or None if no pose."""
        if frame_rgb_u8.dtype != np.uint8:
            raise TypeError("MediaPipePoseEstimator expects uint8 RGB frame")
        if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
            raise ValueError("MediaPipePoseEstimator expects frame shape (H, W, 3)")

        results = self._pose.process(frame_rgb_u8)
        if not results.pose_landmarks:
            return None

        lms: list[PoseLandmark] = []
        for lm in results.pose_landmarks.landmark:
            # visibility is 0..1-ish; treat as score.
            lms.append(PoseLandmark(x=float(lm.x), y=float(lm.y), score=float(lm.visibility)))
        return lms

