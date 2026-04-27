from functools import partial

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    autoML_h2o,
    autoML_sklearn,
    autoML_sklearn_docker,
    autoML_tpot,
    get_dataset,
    save_automl_report,
)
from ..data.node import get_recife_dengue_data

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
                func=autoML_tpot,
                inputs=["X", "y", "params:autoML_tpot", "params:tscv_n_splits"],
                outputs=["tpot_model", "tpot_report"],
                name="autoML_tpot_node",
            ),
            node(
                func=partial(save_automl_report, framework_name="tpot"),
                inputs=["tpot_report", "params:autoML_tpot"],
                outputs="tpot_report_path",
                name="save_tpot_report_node",
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
                func=autoML_h2o,
                inputs=["X", "y", "params:autoML_h2o", "params:tscv_n_splits"],
                outputs=["h2o_model", "h2o_report"],
                name="autoML_h2o_node",
            ),
            node(
                func=partial(save_automl_report, framework_name="h2o"),
                inputs=["h2o_report", "params:autoML_h2o"],
                outputs="h2o_report_path",
                name="save_h2o_report_node",
            ),
        ]
    )

def create_pipeline_autosklearn(**kwargs) -> Pipeline:
    import platform

    if platform.system() == "Windows":
        func_sklearn = autoML_sklearn_docker
    else:
        func_sklearn = autoML_sklearn
    return pipeline(
        [
            node(
                func=get_recife_dengue_data,
                inputs=[],
                outputs=["X", "y"],
                name="get_recife_dengue_data",
            ),
            node(
                func=func_sklearn,
                inputs=["X", "y", "params:autoML_sklearn", "params:tscv_n_splits"],
                outputs=["autosklearn_model", "autosklearn_report"],
                name="autoML_sklearn_node",
            ),
            node(
                func=partial(save_automl_report, framework_name="autosklearn"),
                inputs=["autosklearn_report", "params:autoML_sklearn"],
                outputs="autosklearn_report_path",
                name="save_autosklearn_report_node",
            ),

        ]
    )
