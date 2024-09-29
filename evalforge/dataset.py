"""Golden dataset loading and validation (JSONL or YAML)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS = ("summarize", "next_steps", "triage", "refusal")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    task: str
    input: str
    expected: dict[str, Any]
    severity: int = 1  # 1 (minor) .. 3 (critical)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "task": self.task, "input": self.input, "expected": self.expected,
                "severity": self.severity, "tags": list(self.tags)}


def _parse(raw: dict[str, Any], where: str) -> Case:
    for key in ("id", "task", "input", "expected"):
        if key not in raw:
            raise DatasetError(f"{where}: missing required field '{key}'")
    if raw["task"] not in TASKS:
        raise DatasetError(f"{where}: unknown task '{raw['task']}' (expected one of {TASKS})")
    if not isinstance(raw["expected"], dict) or not raw["expected"]:
        raise DatasetError(f"{where}: 'expected' must be a non-empty mapping")
    severity = int(raw.get("severity", 1))
    if severity not in (1, 2, 3):
        raise DatasetError(f"{where}: severity must be 1, 2 or 3")
    return Case(id=str(raw["id"]), task=raw["task"], input=str(raw["input"]), expected=raw["expected"],
                severity=severity, tags=tuple(raw.get("tags", ())))


def load_cases(path: str | Path) -> list[Case]:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    if path.suffix in (".yaml", ".yml"):
        rows = yaml.safe_load(path.read_text())
        if not isinstance(rows, list):
            raise DatasetError(f"{path}: YAML dataset must be a list of cases")
    else:
        rows = []
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{n}: invalid JSON ({exc})") from exc
    cases = [_parse(r, f"{path.name}[{i}]") for i, r in enumerate(rows)]
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise DatasetError(f"duplicate case ids: {sorted(dupes)}")
    if not cases:
        raise DatasetError(f"{path}: dataset is empty")
    return cases
