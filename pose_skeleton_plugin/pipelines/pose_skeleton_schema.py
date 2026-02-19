"""Pose Skeleton preprocessor pipeline configuration (video mode)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class PoseSkeletonConfig(BasePipelineConfig):
    """Configuration for the Pose Skeleton preprocessor."""

    pipeline_id: ClassVar[str] = "pose-skeleton"
    pipeline_name: ClassVar[str] = "Pose Skeleton"
    pipeline_description: ClassVar[str] = "Render a stick-figure skeleton from pose estimation"
    supports_prompts: ClassVar[bool] = False
    usage: ClassVar[list] = [UsageType.PREPROCESSOR]
    modes: ClassVar[dict] = {"video": ModeDefaults(default=True)}

    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ignore landmarks below this confidence",
        json_schema_extra=ui_field_config(order=1, label="Min Confidence"),
    )
    thickness: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Line thickness for skeleton connections",
        json_schema_extra=ui_field_config(order=2, label="Thickness"),
    )
    joint_radius: int = Field(
        default=3,
        ge=0,
        le=12,
        description="Joint radius for landmark dots",
        json_schema_extra=ui_field_config(order=3, label="Joint Radius"),
    )
    smooth: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="EMA smoothing amount (0 disables)",
        json_schema_extra=ui_field_config(order=4, label="Smooth"),
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
        json_schema_extra=ui_field_config(order=5, label="Debug"),
    )

