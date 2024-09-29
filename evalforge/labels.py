"""Human labels: seeded JSONL file merged with labels collected in the review UI (SQLite wins on conflict)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalforge.models import response_key
from evalforge.store import Store


def load_seed_labels(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        r["response_key"] = response_key(r["case_id"], r["output"])
    return rows


def merged_labels(seed_path: Path, store: Store | None) -> list[dict[str, Any]]:
    """One label per response_key; when several reviewers disagree the majority wins (ties -> fail)."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in load_seed_labels(seed_path) + (store.labels() if store else []):
        by_key.setdefault(r["response_key"], []).append(r)
    merged = []
    for rows in by_key.values():
        # reviewers in the store override a seed label from the same file (later entries are DB labels)
        reviewers = {r.get("reviewer", "seed"): r for r in rows}
        votes = [bool(r["label"]) for r in reviewers.values()]
        merged.append({**rows[-1], "label": sum(votes) * 2 > len(votes)})
    return merged
