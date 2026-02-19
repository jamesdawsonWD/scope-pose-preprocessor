from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.pose_skeleton_pipeline import PoseSkeletonPipeline

    register(PoseSkeletonPipeline)

