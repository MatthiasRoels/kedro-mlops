"""Pydantic schemas to be used in the code-base"""

from pydantic import BaseModel


class MLflowServerConfig(BaseModel):
    tracking_uri: str | None = None

    class Config:
        extra = "forbid"


class MLflowExperiment(BaseModel):
    name: str
    tags: list[str] | None = None


class MLflowRun(BaseModel):
    # if `id` is None, a new run will be created
    id: str | None = None
    # if `name` is None, pipeline name will be used for the run name.
    name: str


class MLflowTrackingConfig(BaseModel):
    enable_tracking: bool = False
    # list of pipelines for which we disable tracking
    exclude_pipelines: list = []
    experiment: MLflowExperiment
    run: MLflowRun | None = None


class MLflowConfig(BaseModel):
    server: MLflowServerConfig
    tracking: MLflowTrackingConfig
