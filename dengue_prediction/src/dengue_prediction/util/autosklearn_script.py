from dengue_prediction.pipelines.autoML.nodes import autoML_sklearn
from dengue_prediction.settings import PROJECT_ROOT
import pandas as pd
import sys
import json
from pathlib import Path

# This script runs inside Docker with the project mounted at /workspace.


def _artifact_reference(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    temp_path = Path(sys.argv[1])
    print("temp_dir =", str(temp_path))

    X = pd.read_csv(temp_path / "temp_autosklearn_X.csv")
    y = pd.read_csv(temp_path / "temp_autosklearn_Y.csv")
    with open(str(temp_path/ "temp_autosklearn_params.json"), "r", encoding="utf-8") as f:
        parms = json.load(f)
    with open(str(temp_path/"temp_autosklearn_n_splits.txt"),"r") as f:
        n_splits = int(f.read())

    automl, result = autoML_sklearn(X, y.squeeze("columns"), params=parms, n_splits=n_splits)

    del automl
    saved_model = result.get("saved_model") or {}
    model_path_value = saved_model.get("model_path")
    if not model_path_value:
        raise RuntimeError("auto-sklearn finished without a saved model artifact.")
    model_path = Path(model_path_value)
    if not model_path.exists():
        raise RuntimeError(f"auto-sklearn reported a missing model artifact: {model_path}")

    model_reference = {
        "artifact_type": "auto-sklearn joblib",
        "model_artifact": _artifact_reference(model_path),
        "container_model_artifact": model_path.as_posix(),
        "load_in": "Docker environment with auto-sklearn installed",
    }
    with open(str(temp_path / "autosklearn_model_reference.json"), "w", encoding="utf-8") as f:
        json.dump(model_reference, f, ensure_ascii=False, indent=2, default=str)

    backend_artifacts = result.setdefault("backend_artifacts", {})
    backend_artifacts["model_path"] = model_reference["model_artifact"]
    backend_artifacts["model_reference"] = model_reference

    with open(str(temp_path/"automl_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
