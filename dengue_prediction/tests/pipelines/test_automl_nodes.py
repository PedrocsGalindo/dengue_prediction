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


class AutoMLNodeTests(unittest.TestCase):
    def test_prepare_tpot_params_translates_legacy_arguments(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tpot import TPOTRegressor
            from tpot.tpot_estimator.estimator import TPOTEstimator

        prepared = nodes._prepare_tpot_params(
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
            nodes._prepare_tpot_params(
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
            nodes._prepare_tpot_params(
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
            nodes._validate_tpot_training_data(X, y)

    def test_prepare_h2o_params_splits_init_arguments(self):
        from h2o.automl import H2OAutoML

        params, init_params, use_leaderboard_frame = nodes._prepare_h2o_params(
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

        X_clean, y_clean, metadata = nodes._prepare_h2o_training_data(X, y)

        self.assertEqual(len(X_clean), 3)
        self.assertEqual(len(y_clean), 3)
        self.assertNotIn("constant", X_clean.columns)
        self.assertFalse(X_clean.select_dtypes(include=[np.number]).isna().any().any())
        self.assertFalse(np.isinf(X_clean.select_dtypes(include=[np.number])).any().any())
        self.assertEqual(metadata["dropped_rows_with_invalid_target"], 1)

    def test_negative_sklearn_scorers_are_treated_as_higher_is_better(self):
        self.assertFalse(nodes._lower_is_better("neg_mean_squared_error"))
        self.assertTrue(nodes._lower_is_better("rmse"))

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

            with patch.object(nodes, "PROJECT_ROOT", project_root):
                model_reference, report = nodes._read_autosklearn_docker_outputs(temp_path)
        finally:
            shutil.rmtree(project_root, ignore_errors=True)

        self.assertEqual(model_reference, str(model_path))
        self.assertEqual(report["backend_artifacts"]["model_path"], str(model_path))
        self.assertEqual(report["backend_artifacts"]["predictions_path"], str(predictions_path))
        self.assertEqual(report["backend_artifacts"]["predictions_row_count"], 1)
        self.assertEqual(report["backend_artifacts"]["predictions_preview"][0]["y_pred"], 1.1)


if __name__ == "__main__":
    unittest.main()
