from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_prediction.settings import DATA_DIR


def save_automl_outputs(
    result: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Save an AutoML result already prepared by tpot.py or h2o.py."""
    output_path = Path(output_dir) if output_dir is not None else _default_output_dir(result)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "report.json"
    search_history_path = output_path / "search_history.json"
    search_history_csv_path = output_path / "search_history.csv"
    best_model_path = output_path / "best_model.json"
    metadata_path = output_path / "metadata.json"

    search_history = result.get("search_history", [])
    best_model = result.get("best_model", {})
    metadata = result.get("metadata", {})

    _write_json(report_path, result)
    _write_json(search_history_path, search_history)
    _write_json(best_model_path, best_model)
    _write_json(metadata_path, metadata)
    _write_csv(search_history_csv_path, search_history)

    return {
        "output_dir": str(output_path),
        "report_json": str(report_path),
        "search_history_json": str(search_history_path),
        "search_history_csv": str(search_history_csv_path),
        "best_model_json": str(best_model_path),
        "metadata_json": str(metadata_path),
    }


def _default_output_dir(result: dict[str, Any]) -> Path:
    metadata = result.get("metadata", {})
    backend = str(metadata.get("backend", "automl"))
    run_id = metadata.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backend_dir = re.sub(r"[^a-z0-9]+", "_", backend.lower()).strip("_") or "automl"
    return DATA_DIR / "results" / "autoML" / backend_dir / str(run_id)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, default=str)


def _write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows if isinstance(rows, list) else []).to_csv(path, index=False)
