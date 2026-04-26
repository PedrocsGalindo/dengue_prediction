import json
import shutil
import unittest
import uuid
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dengue_prediction.pipelines.autoML import nodes
from dengue_prediction.pipelines.autoML.util import common
from dengue_prediction.pipelines.autoML.util import h2o as h2o_util
from dengue_prediction.pipelines.autoML.util import model_saving
from dengue_prediction.pipelines.autoML.util import tpot as tpot_util


class AutoMLNodeTests(unittest.TestCase):
    class FakeEstimator:
        def __init__(self, name="fake", **params):
            self.name = name
            self.params = params or {"alpha": 1}

        def get_params(self, deep=False):
            return self.params

        def predict(self, X):
            return np.zeros(len(X))

        def __str__(self):
            return f"FakeEstimator(name={self.name})"

    class FakePipeline:
        def __init__(self):
            self.steps = [
                ("scale", AutoMLNodeTests.FakeEstimator("StandardScaler")),
                ("model", AutoMLNodeTests.FakeEstimator("Ridge")),
            ]

        def get_params(self, deep=False):
            return {"steps": self.steps}

        def predict(self, X):
            return np.zeros(len(X))

        def __str__(self):
            return "FakePipeline(scale -> model)"

    class FakeTPOT:
        def __init__(self):
            self.fitted_pipeline_ = AutoMLNodeTests.FakePipeline()
            self.evaluated_individuals = pd.DataFrame()

    class FakeH2OModel:
        def __init__(self, model_id):
            self.model_id = model_id

    class FakeH2OLeaderboard:
        def as_data_frame(self):
            return pd.DataFrame(
                [
                    {
                        "model_id": "GBM_1_AutoML_test",
                        "algo": "GBM",
                        "rmse": 1.2,
                        "training_time_ms": 1500,
                    },
                    {
                        "model_id": "GLM_1_AutoML_test",
                        "algo": "GLM",
                        "rmse": 1.5,
                        "training_time_ms": 900,
                    },
                ]
            )

    class FakeH2OAutoML:
        def __init__(self):
            self.leader = AutoMLNodeTests.FakeH2OModel("GBM_1_AutoML_test")
            self.leaderboard = AutoMLNodeTests.FakeH2OLeaderboard()
            self.training_info = {"duration_secs": 2}

    class FakeH2OModule:
        @staticmethod
        def save_model(model, path, force=False):
            model_path = Path(path) / model.model_id
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path.exists() and not force:
                raise RuntimeError("model already exists")
            model_path.write_text("h2o model bytes", encoding="utf-8")
            return str(model_path)

        @staticmethod
        def get_model(model_id):
            return AutoMLNodeTests.FakeH2OModel(model_id)

    def test_prepare_tpot_params_translates_legacy_arguments(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tpot import TPOTRegressor
            from tpot.tpot_estimator.estimator import TPOTEstimator

        prepared = tpot_util._prepare_tpot_params(
            {
                "search_space": "linear-light",
                "scorers": ["neg_mean_squared_error"],
                "scorers_weights": [1],
                "population_size": 8,
                "offspring_size": 8,
                "cv_n_splits": 3,
                "mutation_rate": 0.9,
                "crossover_rate": 0.1,
                "processes": False,
                "n_jobs": 1,
            },
            TPOTRegressor,
            TPOTEstimator,
        )

        self.assertEqual(prepared["cv"], 3)
        self.assertEqual(prepared["mutate_probability"], 0.9)
        self.assertEqual(prepared["crossover_probability"], 0.1)
        self.assertNotIn("cv_n_splits", prepared)
        self.assertNotIn("offspring_size", prepared)
        self.assertNotIn("mutation_rate", prepared)
        self.assertNotIn("crossover_rate", prepared)

    def test_prepare_tpot_params_rejects_unknown_arguments(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tpot import TPOTRegressor
            from tpot.tpot_estimator.estimator import TPOTEstimator

        with self.assertRaisesRegex(ValueError, "Unsupported TPOT 1.1.0 parameters"):
            tpot_util._prepare_tpot_params(
                {"search_space": "linear-light", "unknown_flag": True},
                TPOTRegressor,
                TPOTEstimator,
            )

    def test_prepare_tpot_params_rejects_generation_time_conflict(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tpot import TPOTRegressor
            from tpot.tpot_estimator.estimator import TPOTEstimator

        with self.assertRaisesRegex(ValueError, "either 'generations' or 'max_time_mins'"):
            tpot_util._prepare_tpot_params(
                {"generations": 2, "max_time_mins": 5},
                TPOTRegressor,
                TPOTEstimator,
            )

    def test_validate_tpot_training_data_rejects_non_finite_and_unhelpful_columns(self):
        X = pd.DataFrame(
            {
                "valid": [1.0, 2.0, 3.0],
                "has_inf": [1.0, np.inf, 3.0],
                "constant": [7.0, 7.0, 7.0],
                "duplicate": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([1.0, 2.0, 3.0])

        with self.assertRaisesRegex(
            ValueError,
            "infinite values.*constant feature columns.*duplicated feature columns",
        ):
            tpot_util._validate_tpot_training_data(X, y)

    def test_prepare_h2o_params_splits_init_arguments(self):
        from h2o.automl import H2OAutoML

        params, init_params, use_leaderboard_frame = h2o_util._prepare_h2o_params(
            {
                "max_models": 3,
                "stopping_metric": "RMSE",
                "use_leaderboard_frame": False,
                "init": {"max_mem_size": "1G", "nthreads": 1},
            },
            H2OAutoML,
        )

        self.assertEqual(params, {"max_models": 3, "stopping_metric": "RMSE"})
        self.assertEqual(init_params, {"max_mem_size": "1G", "nthreads": 1})
        self.assertFalse(use_leaderboard_frame)

    def test_prepare_h2o_training_data_cleans_non_finite_and_constant_columns(self):
        X = pd.DataFrame(
            {
                "valid": [1.0, np.nan, 3.0, 4.0],
                "has_inf": [1.0, np.inf, 3.0, 4.0],
                "constant": [7.0, 7.0, 7.0, 7.0],
                "category": ["a", None, "b", "c"],
            }
        )
        y = pd.Series([1.0, 2.0, np.inf, 4.0])

        X_clean, y_clean, metadata = h2o_util._prepare_h2o_training_data(X, y)

        self.assertEqual(len(X_clean), 3)
        self.assertEqual(len(y_clean), 3)
        self.assertNotIn("constant", X_clean.columns)
        self.assertFalse(X_clean.select_dtypes(include=[np.number]).isna().any().any())
        self.assertFalse(np.isinf(X_clean.select_dtypes(include=[np.number])).any().any())
        self.assertEqual(metadata["dropped_rows_with_invalid_target"], 1)

    def test_negative_sklearn_scorers_are_treated_as_higher_is_better(self):
        self.assertFalse(common._lower_is_better("neg_mean_squared_error"))
        self.assertTrue(common._lower_is_better("rmse"))

    def test_reads_autosklearn_docker_outputs_without_unpickling_model(self):
        project_root = Path.cwd() / "temp" / f"autosklearn_output_reader_test_{uuid.uuid4().hex}"
        shutil.rmtree(project_root, ignore_errors=True)
        try:
            temp_path = project_root / "temp"
            temp_path.mkdir(parents=True)
            model_path = project_root / "dengue_prediction" / "data" / "06_models" / "autosklearn_model.pkl"
            predictions_path = (
                project_root
                / "dengue_prediction"
                / "data"
                / "results"
                / "autoML"
                / "low"
                / "autosklearn_predictions.csv"
            )
            model_path.parent.mkdir(parents=True)
            predictions_path.parent.mkdir(parents=True)
            model_path.write_text("docker-only pickle placeholder", encoding="utf-8")
            pd.DataFrame({"row_index": [0], "y_true": [1.0], "y_pred": [1.1]}).to_csv(
                predictions_path,
                index=False,
            )

            result = {
                "backend": "auto-sklearn",
                "preset": "low",
                "backend_artifacts": {
                    "predictions_path": "dengue_prediction/data/results/autoML/low/autosklearn_predictions.csv",
                },
            }
            reference = {
                "model_artifact": "dengue_prediction/data/06_models/autosklearn_model.pkl",
                "artifact_type": "auto-sklearn pickle",
            }
            (temp_path / "automl_result.json").write_text(json.dumps(result), encoding="utf-8")
            (temp_path / "autosklearn_model_reference.json").write_text(
                json.dumps(reference),
                encoding="utf-8",
            )

            with patch.object(common, "PROJECT_ROOT", project_root):
                model_reference, report = common._read_autosklearn_docker_outputs(temp_path)
        finally:
            shutil.rmtree(project_root, ignore_errors=True)

        self.assertEqual(model_reference, str(model_path))
        self.assertEqual(report["backend_artifacts"]["model_path"], str(model_path))
        self.assertEqual(report["backend_artifacts"]["predictions_path"], str(predictions_path))
        self.assertEqual(report["backend_artifacts"]["predictions_row_count"], 1)
        self.assertEqual(report["backend_artifacts"]["predictions_preview"][0]["y_pred"], 1.1)

    def test_save_tpot_model_writes_fold_artifacts_and_metadata(self):
        project_root = Path.cwd() / "temp" / f"tpot_model_save_test_{uuid.uuid4().hex}"
        data_dir = project_root / "dengue_prediction" / "data"
        shutil.rmtree(project_root, ignore_errors=True)
        try:
            with patch.object(model_saving, "DATA_DIR", data_dir):
                metadata = tpot_util.save_tpot_model(
                    self.FakeTPOT(),
                    preset="low",
                    fold=1,
                    training_time=1.5,
                    validation_score=-4.0,
                    metrics={"mae": 1.0, "mse": 4.0, "rmse": 2.0, "r2": 0.8},
                )

            fold_dir = data_dir / "results" / "models" / "autoML" / "low" / "tpot" / "fold_1"
            metadata_path = fold_dir / "metadata.json"

            self.assertTrue(metadata_path.exists())
            self.assertTrue((fold_dir / "pipeline.txt").exists())
            self.assertTrue((fold_dir / "pipeline_steps.json").exists())
            self.assertTrue(Path(metadata["model_path"]).exists())
            self.assertEqual(metadata["backend"], "TPOT")
            self.assertEqual(metadata["fold"], "fold_1")
            self.assertEqual(
                metadata["pipeline_steps"]["final_estimator"]["class_name"],
                "FakeEstimator",
            )
        finally:
            shutil.rmtree(project_root, ignore_errors=True)

    def test_save_h2o_model_writes_leader_and_metadata_without_overwriting(self):
        project_root = Path.cwd() / "temp" / f"h2o_model_save_test_{uuid.uuid4().hex}"
        data_dir = project_root / "dengue_prediction" / "data"
        shutil.rmtree(project_root, ignore_errors=True)
        try:
            with patch.object(model_saving, "DATA_DIR", data_dir):
                metadata = h2o_util.save_h2o_model(
                    self.FakeH2OModule,
                    self.FakeH2OAutoML(),
                    preset="low",
                    fold="final",
                    training_time=2.0,
                    validation_score=1.2,
                    metrics={"mae": 1.0, "mse": 1.44, "rmse": 1.2, "r2": 0.7},
                    save_leaderboard_models=True,
                )

            fold_dir = data_dir / "results" / "models" / "autoML" / "low" / "h2o" / "final"

            self.assertTrue((fold_dir / "metadata.json").exists())
            self.assertTrue((fold_dir / "leaderboard.json").exists())
            self.assertTrue(Path(metadata["model_path"]).exists())
            self.assertEqual(metadata["backend"], "H2O AutoML")
            self.assertEqual(metadata["model_id"], "GBM_1_AutoML_test")
            self.assertEqual(metadata["algo"], "GBM")
            self.assertEqual(len(metadata["intermediate_models"]), 1)
        finally:
            shutil.rmtree(project_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
