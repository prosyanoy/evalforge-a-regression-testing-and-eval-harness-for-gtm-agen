"""MLflow experiment tracking. Works against the embedded SQLite store or a tracking server."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from evalforge.config import Settings, get_settings

log = logging.getLogger("evalforge")


def log_run(summary: dict[str, Any], artifacts: dict[str, str], settings: Settings | None = None) -> str | None:
    """Log a run to MLflow; returns the MLflow run id, or None (with a warning) if tracking is unavailable.
    `artifacts` maps filename -> text content."""
    s = settings or get_settings()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
        mlflow.set_tracking_uri(s.mlflow_uri)
        mlflow.set_experiment(s.mlflow_experiment)
        with mlflow.start_run(run_name=f"{summary['prompt_version']}-{summary['run_id']}") as run:
            mlflow.set_tags({"prompt_version": summary["prompt_version"], "agent": summary.get("agent", ""),
                             "judge": summary.get("judge", ""), "evalforge_run_id": summary["run_id"]})
            mlflow.log_params({"prompt_version": summary["prompt_version"], "n_cases": summary["metrics"]["n_cases"],
                               "judge": summary.get("judge", "")})
            mlflow.log_metrics({k: float(v) for k, v in summary["metrics"].items() if isinstance(v, (int, float))})
            with tempfile.TemporaryDirectory() as tmp:
                for name, content in artifacts.items():
                    p = Path(tmp) / name
                    p.write_text(content)
                    mlflow.log_artifact(str(p))
                (Path(tmp) / "results.json").write_text(json.dumps(summary["results"], indent=2))
                mlflow.log_artifact(str(Path(tmp) / "results.json"))
            return run.info.run_id
    except Exception as exc:  # tracking must never take down an eval run
        log.warning("MLflow logging skipped (%s: %s)", type(exc).__name__, exc)
        return None
