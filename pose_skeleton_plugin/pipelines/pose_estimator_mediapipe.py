"""MediaPipe pose estimator wrapper.

Uses the MediaPipe Tasks API (>=0.10.14) which replaced the legacy
``mp.solutions`` interface.  Supports both IMAGE (single-shot) and
VIDEO (streaming with temporal tracking) running modes.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
from pathlib import Path

import numpy as np

from .pose_skeleton_renderer import PoseLandmark

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "pose_landmarker.task"


def _find_model_path() -> str:
    """Locate the ``.task`` model file.

    Search order:
    1. ``POSE_LANDMARKER_MODEL`` environment variable (explicit override)
    2. Bundled inside the installed package (``pose_skeleton_plugin/models/``)
    3. Relative to this source file (development / editable installs)
    """
    env = os.environ.get("POSE_LANDMARKER_MODEL")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning("[PoseEstimator] POSE_LANDMARKER_MODEL=%s not found", env)

    # Bundled package data (works for normal pip installs)
    try:
        ref = importlib.resources.files("pose_skeleton_plugin") / "models" / _MODEL_FILENAME
        with importlib.resources.as_file(ref) as pkg_path:
            if pkg_path.is_file():
                return str(pkg_path)
    except Exception:
        pass

    # Fallback: relative paths for editable / dev installs
    for p in [
        Path(__file__).resolve().parent / _MODEL_FILENAME,
        Path(__file__).resolve().parent.parent / _MODEL_FILENAME,
        Path(__file__).resolve().parent.parent.parent / _MODEL_FILENAME,
    ]:
        if p.is_file():
            return str(p)

    raise FileNotFoundError(
        f"Cannot find {_MODEL_FILENAME}. Either:\n"
        f"  - place it in pose_skeleton_plugin/models/\n"
        f"  - set the POSE_LANDMARKER_MODEL env var\n"
        f"  - or place it in the project root (dev mode)"
    )


def get_mediapipe_pose_connections() -> list[tuple[int, int]]:
    """Return pose connections as ``(start, end)`` index pairs."""
    from mediapipe.tasks.python.vision import PoseLandmarksConnections

    return [
        (int(c.start), int(c.end))
        for c in PoseLandmarksConnections.POSE_LANDMARKS
    ]


def _extract_landmarks(result) -> list[PoseLandmark] | None:
    """Pull our PoseLandmark list out of a PoseLandmarkerResult."""
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None
    return [
        PoseLandmark(x=float(lm.x), y=float(lm.y), score=float(lm.visibility))
        for lm in result.pose_landmarks[0]
    ]


class MediaPipePoseEstimator:
    """Estimate a single person's pose landmarks from an RGB frame.

    Parameters
    ----------
    model_path:
        Explicit path to a ``pose_landmarker*.task`` model.  When *None*
        the file is located automatically via ``_find_model_path()``.
    streaming:
        If *True* use ``RunningMode.VIDEO`` which maintains temporal state
        across frames (better tracking).  Requires monotonically increasing
        ``timestamp_ms`` in each ``estimate()`` call.
        If *False* (default) use ``RunningMode.IMAGE`` — every frame is
        independent.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        streaming: bool = False,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        resolved = model_path or _find_model_path()
        logger.info("[PoseEstimator] model=%s  streaming=%s", resolved, streaming)

        mode = vision.RunningMode.VIDEO if streaming else vision.RunningMode.IMAGE
        base_options = mp.tasks.BaseOptions(
            model_asset_path=resolved,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._streaming = streaming
        self._frame_idx = 0

    def estimate(
        self,
        frame_rgb_u8: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> list[PoseLandmark] | None:
        """Return landmarks in normalized coords, or *None* if no pose.

        Parameters
        ----------
        frame_rgb_u8:
            ``(H, W, 3)`` uint8 RGB image.
        timestamp_ms:
            Monotonically increasing timestamp — **required** in streaming
            mode.  Ignored in image mode.
        """
        import mediapipe as mp

        if frame_rgb_u8.dtype != np.uint8:
            raise TypeError("MediaPipePoseEstimator expects uint8 RGB frame")
        if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
            raise ValueError("MediaPipePoseEstimator expects frame shape (H, W, 3)")

        frame = np.ascontiguousarray(frame_rgb_u8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        if self._streaming:
            if timestamp_ms is None:
                timestamp_ms = self._frame_idx * 33  # ~30 fps fallback
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        self._frame_idx += 1
        return _extract_landmarks(result)

