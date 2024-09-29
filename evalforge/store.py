"""SQLite persistence for runs, per-case results and human review labels."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, prompt_version TEXT NOT NULL, agent TEXT, judge TEXT,
  created TEXT NOT NULL, metrics_json TEXT NOT NULL, mlflow_run_id TEXT);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT NOT NULL REFERENCES runs(run_id), case_id TEXT NOT NULL, task TEXT, severity INTEGER,
  score REAL, passed INTEGER, output_json TEXT, errors_json TEXT, judge_json TEXT,
  latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  PRIMARY KEY (run_id, case_id));
CREATE TABLE IF NOT EXISTS human_labels (
  response_key TEXT NOT NULL, reviewer TEXT NOT NULL, case_id TEXT NOT NULL, output_json TEXT NOT NULL,
  label INTEGER NOT NULL, note TEXT DEFAULT '', created TEXT NOT NULL,
  PRIMARY KEY (response_key, reviewer));
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---- runs -------------------------------------------------------------------------------------------
    def save_run(self, summary: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)", (
                summary["run_id"], summary["prompt_version"], summary.get("agent"), summary.get("judge"),
                summary["created"], json.dumps(summary["metrics"]), summary.get("mlflow_run_id")))
            self.conn.execute("DELETE FROM results WHERE run_id=?", (summary["run_id"],))
            self.conn.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (summary["run_id"], r["case_id"], r["task"], r["severity"], r["score"], int(r["passed"]),
                 json.dumps(r["output"]), json.dumps(r["errors"]), json.dumps(r["judge"]),
                 r["latency_ms"], r["input_tokens"], r["output_tokens"], r["cost_usd"])
                for r in summary["results"]])

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created, rowid").fetchall()
        return [{**dict(r), "metrics": json.loads(r["metrics_json"])} for r in rows]

    def latest_run_id(self) -> str | None:
        row = self.conn.execute("SELECT run_id FROM runs ORDER BY created DESC, rowid DESC LIMIT 1").fetchone()
        return row["run_id"] if row else None

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise KeyError(f"unknown run {run_id}")
        results = [{"case_id": r["case_id"], "task": r["task"], "severity": r["severity"], "score": r["score"],
                    "passed": bool(r["passed"]), "output": json.loads(r["output_json"]),
                    "errors": json.loads(r["errors_json"]), "judge": json.loads(r["judge_json"]),
                    "latency_ms": r["latency_ms"], "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"], "cost_usd": r["cost_usd"]}
                   for r in self.conn.execute("SELECT * FROM results WHERE run_id=? ORDER BY case_id", (run_id,))]
        return {"run_id": run["run_id"], "prompt_version": run["prompt_version"], "agent": run["agent"],
                "judge": run["judge"], "created": run["created"], "metrics": json.loads(run["metrics_json"]),
                "mlflow_run_id": run["mlflow_run_id"], "results": results}

    # ---- human labels -----------------------------------------------------------------------------------
    def add_label(self, response_key: str, case_id: str, output: dict[str, Any], label: bool,
                  reviewer: str = "human", note: str = "", created: str | None = None) -> None:
        from datetime import datetime, timezone
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO human_labels VALUES (?,?,?,?,?,?,?)", (
                response_key, reviewer, case_id, json.dumps(output), int(label), note,
                created or datetime.now(timezone.utc).isoformat()))

    def labels(self) -> list[dict[str, Any]]:
        return [{"response_key": r["response_key"], "reviewer": r["reviewer"], "case_id": r["case_id"],
                 "output": json.loads(r["output_json"]), "label": bool(r["label"]), "note": r["note"]}
                for r in self.conn.execute("SELECT * FROM human_labels ORDER BY created")]
