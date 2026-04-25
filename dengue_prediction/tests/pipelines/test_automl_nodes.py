import unittest
import warnings

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

    def test_negative_sklearn_scorers_are_treated_as_higher_is_better(self):
        self.assertFalse(nodes._lower_is_better("neg_mean_squared_error"))
        self.assertTrue(nodes._lower_is_better("rmse"))


if __name__ == "__main__":
    unittest.main()
