"""Seed a synthetic "human-labelled" subset for judge calibration demos.

These labels are SYNTHETIC: a lenient-on-near-misses reviewer model (pass iff rule score >= 0.75) with ~12% of
labels flipped deterministically (hash of the response) to mimic human noise. Real labels come from the review UI.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evalforge.agents.reference import VERSIONS, build
from evalforge.dataset import Case
from evalforge.graders import grade_case
from evalforge.models import Request, normalize_response, response_key


def build_seed_labels(cases: list[Case], versions: tuple[str, ...] = ("v1", "v2", "v3"), stride: int = 2) -> list[dict]:
    rows, seen = [], set()
    for v in versions:
        agent = build(v)
        for i, case in enumerate(cases):
            if (i + VERSIONS_INDEX[v]) % stride:
                continue
            out = normalize_response(agent(Request(case.task, case.input)), Request(case.task, case.input)).output
            key = response_key(case.id, out)
            if key in seen:
                continue
            seen.add(key)
            passed = grade_case(case, out).score >= 0.75
            if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100 < 12:
                passed = not passed
            rows.append({"case_id": case.id, "output": out, "label": passed, "reviewer": "seed", "note": "synthetic"})
    return rows


VERSIONS_INDEX = {v: i for i, v in enumerate(VERSIONS)}


def write_seed_labels(cases: list[Case], path: Path) -> int:
    rows = build_seed_labels(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)
