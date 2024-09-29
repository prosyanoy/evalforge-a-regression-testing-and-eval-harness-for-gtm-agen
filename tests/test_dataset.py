import json
import subprocess
import sys

import pytest

from evalforge.config import ROOT
from evalforge.dataset import DatasetError, load_cases


def test_golden_dataset_shape(cases):
    assert len(cases) == 50
    assert {c.task for c in cases} == {"summarize", "next_steps", "triage", "refusal"}
    assert len({c.id for c in cases}) == 50


def test_committed_dataset_matches_generator(tmp_path):
    out = tmp_path / "gen.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_dataset.py"), str(out)], check=True, capture_output=True)
    assert out.read_text() == (ROOT / "data/golden/cases.jsonl").read_text()


def test_yaml_loading(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- {id: a, task: triage, input: hi, expected: {fields: {category: bug}}}\n")
    assert load_cases(p)[0].id == "a"


@pytest.mark.parametrize("row,msg", [
    ({"id": "x", "task": "nope", "input": "i", "expected": {"a": 1}}, "unknown task"),
    ({"id": "x", "task": "triage", "input": "i"}, "missing required"),
    ({"id": "x", "task": "triage", "input": "i", "expected": {"a": 1}, "severity": 9}, "severity"),
])
def test_invalid_rows_rejected(tmp_path, row, msg):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetError, match=msg):
        load_cases(p)


def test_duplicate_ids_rejected(tmp_path):
    row = json.dumps({"id": "x", "task": "triage", "input": "i", "expected": {"fields": {"a": "b"}}})
    p = tmp_path / "c.jsonl"
    p.write_text(f"{row}\n{row}\n")
    with pytest.raises(DatasetError, match="duplicate"):
        load_cases(p)
