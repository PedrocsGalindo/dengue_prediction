from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

# Importamos a função de dados que é comum a ambos
from dengue_prediction.pipelines.data.node import get_recife_dengue_data

# Importamos a função de treino específica do RF que criamos no nodes.py
from .nodes import train_random_forest


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_recife_dengue_data,
                inputs=[],
                outputs=["X", "y"],
                name="get_recife_dengue_data_for_random_forest",
            ),
            node(
                func=train_random_forest,
                inputs=["X", "y", "params:random_forest"], # Referência aos novos parâmetros
                outputs=[
                    "random_forest_model",
                    "random_forest_report",
                    "random_forest_predictions",
                ],
                name="train_random_forest_node",
            ),
        ]
    )