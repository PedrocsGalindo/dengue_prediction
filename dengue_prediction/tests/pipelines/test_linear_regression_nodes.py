import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dengue_prediction.pipelines.linear_regression import nodes


class LinearRegressionNodeTests(unittest.TestCase):
    def test_train_linear_regression_returns_metrics_and_artifacts(self):
        X = pd.DataFrame(
            {
                "temperatura_ar_media_c": range(30),
                "umidade_relativa_media": [70 + (index % 5) for index in range(30)],
                "mes": [1 + (index // 10) for index in range(30)],
            }
        )
        y = pd.Series(
            [2 * row + 0.5 * (index % 5) for index, row in enumerate(range(30))],
            name="casos_dengue",
        )
        data_dir = Path.cwd() / "temp" / f"linear_regression_test_{uuid.uuid4().hex}"
        shutil.rmtree(data_dir, ignore_errors=True)

        try:
            with patch.object(nodes, "DATA_DIR", data_dir):
                model, report, predictions = nodes.train_linear_regression(
                    X,
                    y,
                    {
                        "test_size": 0.2,
                        "n_splits": 3,
                        "fit_intercept": True,
                        "positive": False,
                        "scale_features": True,
                    },
                )

            self.assertEqual(report["metadata"]["model_name"], "LinearRegression")
            self.assertEqual(report["data"]["train_rows"], 24)
            self.assertEqual(report["data"]["test_rows"], 6)
            self.assertIn("rmse", report["test_metrics"])
            self.assertEqual(len(report["cross_validation"]["fold_metrics"]), 3)
            self.assertEqual(list(predictions.columns), ["actual", "predicted", "residual"])
            self.assertTrue(Path(report["artifacts"]["model_path"]).exists())
            self.assertTrue(Path(report["artifacts"]["report_path"]).exists())
            self.assertTrue(Path(report["artifacts"]["predictions_path"]).exists())
            saved_report = json.loads(Path(report["artifacts"]["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_report["artifacts"], report["artifacts"])
            self.assertEqual(model.named_steps["model"].__class__.__name__, "LinearRegression")
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
