"""Pose Skeleton pipeline — renders a pose stick figure from video frames."""

from __future__ import annotations

import logging
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

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
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
            raise TypeError("Expected video frames to be torch tensors")

        frame = frame_t.squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (1,H,W,3) -> (H,W,3); got {frame.shape}")

        H, W = int(frame.shape[0]), int(frame.shape[1])
        frame_u8 = np.clip(frame, 0.0, 255.0).astype(np.uint8, copy=False)

        estimator = self._get_estimator()
        landmarks = estimator.estimate(frame_u8) if estimator is not None else None

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

        if debug:
            nz = int(np.count_nonzero(out_np[..., 0] > 0))
            n_lm = 0 if not smoothed else len(smoothed)
            logger.info(
                "[PoseSkeleton] in=%s range=[%.1f,%.1f] landmarks=%d nonzero=%d out=[%.3f,%.3f] "
                "params(min_conf=%.2f thick=%d joint=%d smooth=%.2f)",
                tuple(frame_t.shape),
                float(frame.min()),
                float(frame.max()),
                n_lm,
                nz,
                float(out_np.min()),
                float(out_np.max()),
                min_confidence,
                thickness,
                joint_radius,
                smooth,
            )

        return {"video": out}

    def _get_estimator(self):
        if self._estimator is not None:
            return self._estimator

        try:
            from .pose_estimator_mediapipe import MediaPipePoseEstimator
        except Exception as e:  # pragma: no cover (only when deps missing)
            raise RuntimeError(
                "MediaPipe is required for PoseSkeletonPipeline. "
                "Install the plugin dependencies (mediapipe, opencv-python-headless)."
            ) from e

        self._estimator = MediaPipePoseEstimator()
        return self._estimator

    def _get_connections(self) -> list[tuple[int, int]]:
        if self._connections is not None:
            return self._connections

        try:
            from .pose_estimator_mediapipe import get_mediapipe_pose_connections

            self._connections = get_mediapipe_pose_connections()
        except Exception:
            # Minimal fallback: a simple stick line for safety.
            self._connections = [(0, 1)]
        return self._connections

