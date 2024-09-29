"""Regression gate: compare a run against a committed baseline and decide pass/fail."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalforge.config import Settings, get_settings
from evalforge.dataset import TASKS


def make_baseline(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact, diff-friendly snapshot suitable for committing."""
    return {"prompt_version": summary["prompt_version"], "created": summary["created"], "judge": summary.get("judge"),
            "metrics": {k: round(v, 6) for k, v in summary["metrics"].items()},
            "cases": {r["case_id"]: {"passed": r["passed"], "score": r["score"], "severity": r["severity"]}
                      for r in summary["results"]}}


def save_baseline(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_baseline(summary), indent=2, sort_keys=True) + "\n")


def compare(current: dict[str, Any], baseline: dict[str, Any], settings: Settings | None = None,
            max_score_drop: float | None = None) -> dict[str, Any]:
    """`current` is a run summary; `baseline` a make_baseline() snapshot. Drops are absolute (0..1 scale)."""
    s = settings or get_settings()
    cur_m, base_m = current["metrics"], baseline["metrics"]
    limits = [("mean_score", max_score_drop if max_score_drop is not None else s.max_score_drop),
              ("pass_rate", s.max_pass_rate_drop), ("judge_score", s.max_judge_drop)]
    limits += [(f"pass_rate.{t}", s.max_task_pass_drop) for t in TASKS if f"pass_rate.{t}" in base_m]
    checks = []
    for name, limit in limits:
        if name not in cur_m:
            continue
        drop = base_m[name] - cur_m[name]
        checks.append({"name": name, "baseline": base_m[name], "current": cur_m[name], "drop": drop,
                       "limit": limit, "ok": drop <= limit + 1e-9})
    cur_cases = {r["case_id"]: r for r in current["results"]}
    common = [cid for cid in baseline["cases"] if cid in cur_cases]
    new_crit = [cid for cid in common if baseline["cases"][cid]["severity"] == 3
                and baseline["cases"][cid]["passed"] and not cur_cases[cid]["passed"]]
    regressions = [cid for cid in common if baseline["cases"][cid]["passed"] and not cur_cases[cid]["passed"]]
    fixes = [cid for cid in common if not baseline["cases"][cid]["passed"] and cur_cases[cid]["passed"]]
    checks.append({"name": "new_critical_failures", "baseline": 0, "current": len(new_crit), "drop": len(new_crit),
                   "limit": 0, "ok": not new_crit})
    return {"passed": all(c["ok"] for c in checks), "checks": checks, "regressed_cases": regressions,
            "new_critical_failures": new_crit, "fixed_cases": fixes,
            "missing_from_run": sorted(set(baseline["cases"]) - set(cur_cases)),
            "baseline_version": baseline["prompt_version"], "current_version": current["prompt_version"]}


def render_gate_md(res: dict[str, Any]) -> str:
    status = "PASSED" if res["passed"] else "FAILED"
    lines = [f"## Regression gate: {status}", "",
             f"Baseline `{res['baseline_version']}` vs current `{res['current_version']}`", "",
             "| check | baseline | current | drop | allowed | result |", "|---|---|---|---|---|---|"]
    for c in res["checks"]:
        fmt = (lambda v: f"{v:.3f}") if c["name"] != "new_critical_failures" else (lambda v: str(int(v)))
        lines.append(f"| {c['name']} | {fmt(c['baseline'])} | {fmt(c['current'])} | {c['drop']:+.3f} | "
                     f"{c['limit']:.3f} | {'ok' if c['ok'] else '**FAIL**'} |")
    if res["regressed_cases"]:
        lines += ["", f"Regressed cases (passed in baseline, fail now): {', '.join(res['regressed_cases'])}"]
    if res["fixed_cases"]:
        lines += ["", f"Newly fixed cases: {', '.join(res['fixed_cases'])}"]
    if res["missing_from_run"]:
        lines += ["", f"Baseline cases not present in this run (ignored): {', '.join(res['missing_from_run'])}"]
    return "\n".join(lines) + "\n"
