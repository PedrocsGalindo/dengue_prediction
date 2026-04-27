from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from dengue_prediction.pipelines.data.node import get_recife_dengue_data

from .h2o import run_h2o_automl
from .tpot import run_tpot_automl


def create_pipeline(**kwargs) -> Pipeline:
    return create_pipeline_tpot(**kwargs)


def create_pipeline_tpot(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_recife_dengue_data,
                inputs=[],
                outputs=["X", "y"],
                name="get_recife_dengue_data",
            ),
            node(
                func=run_tpot_automl,
                inputs=["X", "y", "params:autoML_tpot", "params:tscv_n_splits"],
                outputs=["tpot_model", "tpot_report"],
                name="train_tpot",
            ),
        ]
    )


def create_pipeline_h2o(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_recife_dengue_data,
                inputs=[],
                outputs=["X", "y"],
                name="get_recife_dengue_data",
            ),
            node(
                func=run_h2o_automl,
                inputs=["X", "y", "params:autoML_h2o", "params:tscv_n_splits"],
                outputs=["h2o_model", "h2o_report", "h2o_leaderboard"],
                name="train_h2o",
            ),
        ]
    )
