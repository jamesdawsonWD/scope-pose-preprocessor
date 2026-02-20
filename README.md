# scope-plugin-pose-skeleton

A [Scope](https://github.com/daydreamlive/scope) **preprocessor** plugin that converts incoming video frames into pose stick-figure control images using MediaPipe pose estimation.

## Features

- Real-time pose estimation via [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- Clean skeleton rendering (white joints and limbs on a black background)
- Temporal tracking with optional EMA smoothing for stable output
- Configurable joint radius, line thickness, and confidence threshold

## Install

Follow the [Scope plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md) to install this plugin using the URL:

```
https://github.com/daydreamlive/scope-plugin-pose-skeleton.git
```

## Upgrade

Follow the [Scope plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md) to upgrade this plugin to the latest version.

## Pipeline: Pose Skeleton (`pose-skeleton`)

A **video-mode** preprocessor that takes a single input frame and outputs a rendered skeleton image.

- **Input** — `video=[frame]` where `frame` is `(1, H, W, 3)` in `[0, 255]`
- **Output** — `{"video": out}` where `out` is `(1, H, W, 3)` float in `[0, 1]`

### Parameters

| Parameter | Range | Default | Description |
|---|---|---|---|
| `min_confidence` | 0.0 – 1.0 | 0.5 | Ignore landmarks below this confidence |
| `thickness` | 1 – 12 | 4 | Line thickness for skeleton connections |
| `joint_radius` | 0 – 12 | 3 | Radius for landmark dots |
| `smooth` | 0.0 – 1.0 | 0.0 | EMA smoothing amount (0 disables) |
| `debug` | bool | false | Enable verbose logging |

## Development

```bash
uv venv -p python3.13 .venv
uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```
