# scope-plugin-pose-skeleton

A Scope **preprocessor** plugin that converts incoming video frames into a clean **pose stick-figure** control image.

## What it does

- **Input**: `video=[frame]` where `frame` is `(1, H, W, 3)` in `[0, 255]`
- **Output**: `{"video": out}` where `out` is `(1, H, W, 3)` float in `[0, 1]`
- **Rendering**: black background, white joints/lines

## Pipelines

### Pose Skeleton (`pose-skeleton`)

Runtime params:

- `min_confidence` (0–1)
- `thickness` (1–12)
- `joint_radius` (0–12)
- `smooth` (0–1): EMA smoothing amount
- `debug` (bool)

## Dev

```bash
uv venv -p python3.13 .venv
uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

