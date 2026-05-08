"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from dengue_prediction.pipelines.autoML.pipeline import (
    create_pipeline_h2o,
    create_pipeline_tpot,
    create_pipeline_tpot_ccs,
    create_pipeline_h2o_ccs,
)
from dengue_prediction.pipelines.linear_regression.pipeline import (
    create_pipeline as create_pipeline_linear_regression,
)


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": create_pipeline_tpot(),
        "tpot": create_pipeline_tpot(),
        "h2o": create_pipeline_h2o(),
        "tpot_ccs": create_pipeline_tpot_ccs(),
        "h2o_ccs": create_pipeline_h2o_ccs(),
        "linear_regression": create_pipeline_linear_regression(),
    }
