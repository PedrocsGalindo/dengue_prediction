"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from dengue_prediction.pipelines.autoML.pipeline import (
    create_pipeline_h2o,
    create_pipeline_tpot,
)


def register_pipelines() -> dict[str, Pipeline]:
    tpot_pipeline = create_pipeline_tpot()
    return {
        "__default__": tpot_pipeline,
        "tpot": tpot_pipeline,
        "h2o": create_pipeline_h2o(),
    }
