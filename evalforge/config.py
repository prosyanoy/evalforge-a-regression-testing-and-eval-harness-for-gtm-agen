"""Environment-driven configuration with safe local defaults (no secrets required)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader; real environment variables always win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


def _float(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    dataset: Path
    db: Path
    baseline: Path
    reports: Path
    human_labels: Path
    mlflow_uri: str
    mlflow_experiment: str
    judge: str
    ollama_host: str
    ollama_model: str
    ollama_timeout: float
    price_in_per_1m: float
    price_out_per_1m: float
    max_score_drop: float
    max_pass_rate_drop: float
    max_judge_drop: float
    max_task_pass_drop: float


def get_settings() -> Settings:
    """Read settings fresh from the environment (so tests can monkeypatch)."""
    return Settings(
        dataset=_path("EVALFORGE_DATASET", "data/golden/cases.jsonl"),
        db=_path("EVALFORGE_DB", "data/evalforge.db"),
        baseline=_path("EVALFORGE_BASELINE", "baselines/baseline.json"),
        reports=_path("EVALFORGE_REPORTS", "reports"),
        human_labels=_path("EVALFORGE_HUMAN_LABELS", "data/human_labels.jsonl"),
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'data' / 'mlflow.db'}",
        mlflow_experiment=os.environ.get("MLFLOW_EXPERIMENT", "evalforge"),
        judge=os.environ.get("EVALFORGE_JUDGE", "auto").lower(),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_timeout=_float("OLLAMA_TIMEOUT", 60.0),
        price_in_per_1m=_float("EVALFORGE_PRICE_IN_PER_1M", 0.50),
        price_out_per_1m=_float("EVALFORGE_PRICE_OUT_PER_1M", 1.50),
        max_score_drop=_float("EVALFORGE_MAX_SCORE_DROP", 0.02),
        max_pass_rate_drop=_float("EVALFORGE_MAX_PASS_RATE_DROP", 0.04),
        max_judge_drop=_float("EVALFORGE_MAX_JUDGE_DROP", 0.05),
        max_task_pass_drop=_float("EVALFORGE_MAX_TASK_PASS_DROP", 0.15),
    )
