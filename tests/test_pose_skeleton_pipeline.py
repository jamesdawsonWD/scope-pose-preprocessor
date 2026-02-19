import torch

from pose_skeleton_plugin.pipelines.pose_skeleton_pipeline import PoseSkeletonPipeline
from pose_skeleton_plugin.pipelines.pose_skeleton_renderer import PoseLandmark


class DummyEstimator:
    def __init__(self, landmarks):
        self._landmarks = landmarks

    def estimate(self, frame_rgb_u8):
        return self._landmarks


def test_pipeline_output_contract_shape_and_range():
    # Input: (1, H, W, 3) in [0,255]
    frame = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.float32)
    estimator = DummyEstimator(
        [PoseLandmark(x=0.25, y=0.25, score=1.0), PoseLandmark(x=0.75, y=0.75, score=1.0)]
    )
    pipe = PoseSkeletonPipeline(estimator=estimator, connections=[(0, 1)])
    result = pipe(video=[frame], min_confidence=0.0, thickness=2, joint_radius=2, smooth=0.0)
    out = result["video"]
    assert out.shape == (1, 64, 64, 3)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_pipeline_empty_landmarks_returns_black():
    frame = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.float32)
    estimator = DummyEstimator(None)
    pipe = PoseSkeletonPipeline(estimator=estimator, connections=[(0, 1)])
    out = pipe(video=[frame], min_confidence=0.0, thickness=2, joint_radius=2, smooth=0.0)["video"]
    assert torch.count_nonzero(out) == 0

