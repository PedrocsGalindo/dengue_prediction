from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from dengue_prediction.pipelines.data.node import get_recife_dengue_data

from .nodes import train_linear_regression


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_recife_dengue_data,
                inputs=[],
                outputs=["X", "y"],
                name="get_recife_dengue_data_for_linear_regression",
            ),
            node(
                func=train_linear_regression,
                inputs=["X", "y", "params:linear_regression"],
                outputs=[
                    "linear_regression_model",
                    "linear_regression_report",
                    "linear_regression_predictions",
                ],
                name="train_linear_regression",
            ),
        ]
    )

