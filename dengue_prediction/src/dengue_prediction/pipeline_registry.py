"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from dengue_prediction.pipelines.autoML.pipeline import (  
	create_pipeline_tpot as tpot_pipeline,  
	create_pipeline_h2o as h2o_pipeline,
    create_pipeline_autosklearn as autosklearn_pipeline,
)  


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {  
		"tpot": tpot_pipeline(),  
		"h2o": h2o_pipeline(),  
		"autosklearn": autosklearn_pipeline(), 
	}
