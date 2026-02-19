import numpy as np

from pose_skeleton_plugin.pipelines.pose_skeleton_renderer import (
    LandmarkSmoother,
    PoseLandmark,
    render_skeleton,
)


def test_render_output_shape_and_range():
    H, W = 64, 64
    landmarks = [
        PoseLandmark(x=0.25, y=0.25, score=1.0),
        PoseLandmark(x=0.75, y=0.75, score=1.0),
    ]
    out = render_skeleton(
        height=H,
        width=W,
        landmarks=landmarks,
        connections=[(0, 1)],
        min_confidence=0.0,
        thickness=2,
        joint_radius=2,
    )
    assert out.shape == (H, W, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_empty_landmarks_returns_black():
    H, W = 32, 32
    out = render_skeleton(
        height=H,
        width=W,
        landmarks=None,
        connections=[(0, 1)],
        min_confidence=0.0,
        thickness=2,
        joint_radius=2,
    )
    assert np.all(out == 0.0)


def test_thickness_affects_nonzero_pixel_count():
    H, W = 64, 64
    landmarks = [
        PoseLandmark(x=0.1, y=0.5, score=1.0),
        PoseLandmark(x=0.9, y=0.5, score=1.0),
    ]
    thin = render_skeleton(
        height=H,
        width=W,
        landmarks=landmarks,
        connections=[(0, 1)],
        min_confidence=0.0,
        thickness=1,
        joint_radius=1,
    )
    thick = render_skeleton(
        height=H,
        width=W,
        landmarks=landmarks,
        connections=[(0, 1)],
        min_confidence=0.0,
        thickness=9,
        joint_radius=1,
    )
    thin_nz = int(np.count_nonzero(thin[..., 0] > 0))
    thick_nz = int(np.count_nonzero(thick[..., 0] > 0))
    assert thick_nz > thin_nz


def test_ema_smoothing_moves_less_than_raw():
    smoother = LandmarkSmoother(alpha=0.5)
    prev = [PoseLandmark(x=0.2, y=0.2, score=1.0)]
    nxt = [PoseLandmark(x=0.8, y=0.8, score=1.0)]

    sm0 = smoother.update(prev)
    sm1 = smoother.update(nxt)

    raw_dx = abs(nxt[0].x - prev[0].x)
    sm_dx = abs(sm1[0].x - sm0[0].x)
    assert sm_dx < raw_dx

