"""Run any agent function against the golden dataset and score it."""
from __future__ import annotations

import importlib
import inspect
import statistics
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from evalforge.config import Settings, get_settings
from evalforge.dataset import TASKS, Case
from evalforge.graders import grade_case
from evalforge.graders.judge import Judge
from evalforge.models import Request, Verdict, normalize_response

DEFAULT_AGENT = "evalforge.agents.reference:build"


def resolve_agent(spec: str, prompt_version: str) -> Callable[[Request], Any]:
    """`pkg.module:attr`. If attr takes a `prompt_version` parameter it is a factory, else the agent itself."""
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("agent spec must look like 'package.module:callable'")
    obj = getattr(importlib.import_module(module), attr)
    if "prompt_version" in inspect.signature(obj).parameters:
        return obj(prompt_version=prompt_version)
    return obj


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results)
    m: dict[str, float] = {
        "n_cases": n,
        "mean_score": sum(r["score"] for r in results) / n,
        "pass_rate": sum(r["passed"] for r in results) / n,
        "judge_score": sum((r["judge"]["overall"] - 1) / 4 for r in results) / n,  # normalised to 0..1
        "judge_pass_rate": sum(r["judge"]["passed"] for r in results) / n,
        "critical_failures": sum((not r["passed"]) and r["severity"] == 3 for r in results),
        "latency_p50_ms": statistics.median(r["latency_ms"] for r in results),
        "latency_p95_ms": _pct([r["latency_ms"] for r in results], 0.95),
        "tokens_in_total": sum(r["input_tokens"] for r in results),
        "tokens_out_total": sum(r["output_tokens"] for r in results),
        "cost_usd_total": sum(r["cost_usd"] for r in results),
    }
    for task in TASKS:
        rows = [r for r in results if r["task"] == task]
        if rows:
            m[f"pass_rate.{task}"] = sum(r["passed"] for r in rows) / len(rows)
            m[f"score.{task}"] = sum(r["score"] for r in rows) / len(rows)
    return m


def run_eval(agent: Callable[[Request], Any], cases: list[Case], judge: Judge, prompt_version: str,
             agent_name: str = "", settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    results = []
    for case in cases:
        req = Request(case.task, case.input)
        t0 = time.perf_counter()
        try:
            resp = normalize_response(agent(req), req)
            output, error = resp.output, None
        except Exception as exc:  # a crashing agent is a failed case, not a crashed eval
            resp = normalize_response({"text": ""}, req)
            output, error = {"text": ""}, f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000
        if error:
            from evalforge.models import GradeResult
            grade = GradeResult("agent", 0.0, False, [{"type": "agent_exception", "detail": error}])
            verdict = Verdict({"accuracy": 1, "completeness": 1, "safety": 1, "concision": 1}, 1.0, False,
                              "agent raised an exception", "n/a")
        else:
            grade = grade_case(case, output)
            verdict = judge.judge(case, output)
        cost = (resp.input_tokens * s.price_in_per_1m + resp.output_tokens * s.price_out_per_1m) / 1e6
        results.append({
            "case_id": case.id, "task": case.task, "severity": case.severity, "input": case.input,
            "output": output, "score": round(grade.score, 4), "passed": grade.passed, "errors": grade.errors,
            "judge": {"scores": verdict.scores, "overall": verdict.overall, "passed": verdict.passed,
                      "rationale": verdict.rationale, "backend": verdict.backend},
            "latency_ms": round(latency_ms, 4), "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens, "cost_usd": cost})
    now = datetime.now(timezone.utc)
    return {"run_id": f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}", "prompt_version": prompt_version,
            "agent": agent_name, "judge": judge.name, "created": now.isoformat(),
            "metrics": compute_metrics(results), "results": results}
