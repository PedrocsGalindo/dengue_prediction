from dengue_prediction.pipelines.autoML.nodes import autoML_sklearn
import pandas as pd
import sys
import json
import joblib
from pathlib import Path
## This script is made to run on a dockr conteiner where project/path/dengue_prediction:/workspace


if __name__ == "__main__":
    temp_path = Path(sys.argv[1])
    print("temp_dir =", str(temp_path))
    # load data and params 
    X = pd.read_csv(temp_path / "temp_autosklearn_X.csv")
    y = pd.read_csv(temp_path / "temp_autosklearn_Y.csv")
    with open(str(temp_path/ "temp_autosklearn_params.json"), "r", encoding="utf-8") as f:
        parms = json.load(f)
    with open(str(temp_path/"temp_autosklearn_n_splits.txt"),"r") as f:
        n_splits = int(f.read())

    automl, result = autoML_sklearn(X, y.squeeze("columns"), params=parms, n_splits=n_splits)
    
    # Saving resultr to principal pipeline uses 
    joblib.dump(automl, str(temp_path/"automl.pkl"))
    import json
    with open(str(temp_path/"automl_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
