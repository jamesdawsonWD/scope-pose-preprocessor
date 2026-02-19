"""Pose Skeleton pipeline — renders a pose stick figure from video frames."""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .pose_skeleton_renderer import LandmarkSmoother, render_skeleton
from .pose_skeleton_schema import PoseSkeletonConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class PoseSkeletonPipeline(Pipeline):
    """Preprocessor that outputs a rendered skeleton control image."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return PoseSkeletonConfig

    def __init__(self, *, estimator=None, connections=None, **kwargs):
        self._estimator = estimator
        self._connections = list(connections) if connections is not None else None
        self._smoother: LandmarkSmoother | None = None
        self._smoother_alpha: float | None = None
        self._frame_count = 0
        logger.warning("[PoseSkeleton] __init__ called, pipeline created")

    def prepare(self, **kwargs) -> Requirements:
        logger.warning("[PoseSkeleton] prepare() called")
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        self._frame_count += 1
        try:
            return self._process(kwargs)
        except Exception:
            logger.error("[PoseSkeleton] CRASH in __call__:\n%s", traceback.format_exc())
            raise

    def _process(self, kwargs: dict) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for PoseSkeletonPipeline")
        if not isinstance(video, (list, tuple)) or len(video) < 1:
            raise ValueError("Input video must be a non-empty list of frames")

        min_confidence = float(kwargs.get("min_confidence", 0.5))
        thickness = int(kwargs.get("thickness", 4))
        joint_radius = int(kwargs.get("joint_radius", 3))
        smooth = float(kwargs.get("smooth", 0.0))
        debug = bool(kwargs.get("debug", False))

        frame_t = video[0]
        if not isinstance(frame_t, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(frame_t)}")

        frame = frame_t.squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (1,H,W,3) -> (H,W,3); got {frame.shape}")

        H, W = int(frame.shape[0]), int(frame.shape[1])

        # Auto-detect range: if max <= 1.0 the frame is already normalised → scale up.
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax <= 1.0 + 1e-3:
            frame_for_mp = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frame_for_mp = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        if self._frame_count <= 3 or debug:
            logger.warning(
                "[PoseSkeleton] frame#%d shape=%s dtype=%s range=[%.2f,%.2f] "
                "frame_for_mp range=[%d,%d]",
                self._frame_count,
                tuple(frame_t.shape),
                frame_t.dtype,
                fmin,
                fmax,
                int(frame_for_mp.min()),
                int(frame_for_mp.max()),
            )

        estimator = self._get_estimator()
        timestamp_ms = self._frame_count * 33  # monotonic ~30 fps fallback
        landmarks = estimator.estimate(frame_for_mp, timestamp_ms=timestamp_ms) if estimator is not None else None

        if self._frame_count <= 3 or debug:
            n_lm = 0 if not landmarks else len(landmarks)
            logger.warning("[PoseSkeleton] frame#%d landmarks=%d", self._frame_count, n_lm)

        if smooth <= 0.0:
            if self._smoother is not None:
                self._smoother.update(None)
            smoothed = landmarks
        else:
            alpha = float(np.clip(smooth, 0.0, 1.0))
            if self._smoother is None or self._smoother_alpha != alpha:
                self._smoother = LandmarkSmoother(alpha=alpha)
                self._smoother_alpha = alpha
            smoothed = self._smoother.update(landmarks)

        connections = self._get_connections()
        out_np = render_skeleton(
            height=H,
            width=W,
            landmarks=smoothed,
            connections=connections,
            min_confidence=min_confidence,
            thickness=thickness,
            joint_radius=joint_radius,
        )

        out = torch.from_numpy(out_np).unsqueeze(0)  # (1,H,W,3) float32 in [0,1]

        if self._frame_count <= 3 or debug:
            nz = int(np.count_nonzero(out_np[..., 0] > 0))
            logger.warning(
                "[PoseSkeleton] frame#%d nonzero=%d out=[%.3f,%.3f] out.shape=%s",
                self._frame_count,
                nz,
                float(out_np.min()),
                float(out_np.max()),
                tuple(out.shape),
            )

        return {"video": out}

    def _get_estimator(self):
        if self._estimator is not None:
            return self._estimator

        try:
            from .pose_estimator_mediapipe import MediaPipePoseEstimator
        except Exception:
            logger.error(
                "[PoseSkeleton] Failed to import MediaPipe:\n%s", traceback.format_exc()
            )
            raise RuntimeError(
                "MediaPipe is required for PoseSkeletonPipeline. "
                "Install the plugin dependencies (mediapipe, opencv-python-headless)."
            )

        self._estimator = MediaPipePoseEstimator(streaming=True)
        logger.warning("[PoseSkeleton] MediaPipe estimator created (streaming mode)")
        return self._estimator

    def _get_connections(self) -> list[tuple[int, int]]:
        if self._connections is not None:
            return self._connections

        try:
            from .pose_estimator_mediapipe import get_mediapipe_pose_connections

            self._connections = get_mediapipe_pose_connections()
            logger.warning(
                "[PoseSkeleton] Loaded %d connections from MediaPipe", len(self._connections)
            )
        except Exception:
            logger.error(
                "[PoseSkeleton] Failed to get connections, using fallback:\n%s",
                traceback.format_exc(),
            )
            self._connections = [(0, 1)]
        return self._connections

