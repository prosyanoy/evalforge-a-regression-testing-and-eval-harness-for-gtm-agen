import pytest

from evalforge.config import ROOT
from evalforge.dataset import load_cases


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own SQLite/MLflow/report locations; real data files are only read."""
    monkeypatch.setenv("EVALFORGE_DB", str(tmp_path / "ef.db"))
    monkeypatch.setenv("EVALFORGE_REPORTS", str(tmp_path / "reports"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("EVALFORGE_JUDGE", "rule")
    monkeypatch.setenv("EVALFORGE_BASELINE", str(tmp_path / "baseline.json"))


@pytest.fixture(scope="session")
def cases():
    return load_cases(ROOT / "data/golden/cases.jsonl")
