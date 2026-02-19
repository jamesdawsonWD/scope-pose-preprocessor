"""MediaPipe pose estimator wrapper.

Uses the MediaPipe Tasks API (>=0.10.14) which replaced the legacy
``mp.solutions`` interface.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .pose_skeleton_renderer import PoseLandmark

_MODEL_FILENAME = "pose_landmarker.task"


def _find_model_path() -> str:
    """Locate the .task model file next to this package or in the project root."""
    candidates = [
        Path(__file__).resolve().parent / _MODEL_FILENAME,
        Path(__file__).resolve().parent.parent / _MODEL_FILENAME,
        Path(__file__).resolve().parent.parent.parent / _MODEL_FILENAME,
    ]
    env = os.environ.get("POSE_LANDMARKER_MODEL")
    if env:
        candidates.insert(0, Path(env))
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"Cannot find {_MODEL_FILENAME}. Place it next to the package, in the "
        "project root, or set the POSE_LANDMARKER_MODEL env var."
    )


def get_mediapipe_pose_connections() -> list[tuple[int, int]]:
    """Return pose connections as ``(start, end)`` index pairs."""
    from mediapipe.tasks.python.vision import PoseLandmarksConnections

    return [
        (int(c.start), int(c.end))
        for c in PoseLandmarksConnections.POSE_LANDMARKS
    ]


class MediaPipePoseEstimator:
    """Estimate a single person's pose landmarks from an RGB frame."""

    def __init__(self, model_path: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        resolved = model_path or _find_model_path()

        base_options = mp.tasks.BaseOptions(
            model_asset_path=resolved,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame_rgb_u8: np.ndarray) -> list[PoseLandmark] | None:
        """Return landmarks in normalized coords, or *None* if no pose."""
        import mediapipe as mp

        if frame_rgb_u8.dtype != np.uint8:
            raise TypeError("MediaPipePoseEstimator expects uint8 RGB frame")
        if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
            raise ValueError("MediaPipePoseEstimator expects frame shape (H, W, 3)")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_u8)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        lms: list[PoseLandmark] = []
        for lm in result.pose_landmarks[0]:
            lms.append(
                PoseLandmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    score=float(lm.visibility),
                )
            )
        return lms

