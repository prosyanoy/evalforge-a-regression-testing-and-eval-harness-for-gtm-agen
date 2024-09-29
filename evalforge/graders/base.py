"""Pluggable grader registry.

A grader is `fn(case, output) -> GradeResult | None`; returning None means "not applicable to this case"
(its expectation keys are absent). Register custom graders with `@register_grader("name")` and list them
in a case's `graders:` field, or rely on the defaults (all registered rule graders).
"""
from __future__ import annotations

from typing import Any, Callable

from evalforge.dataset import Case
from evalforge.models import GradeResult

Grader = Callable[[Case, dict[str, Any]], "GradeResult | None"]
GRADERS: dict[str, Grader] = {}


def register_grader(name: str) -> Callable[[Grader], Grader]:
    def deco(fn: Grader) -> Grader:
        GRADERS[name] = fn
        return fn
    return deco


def grade_case(case: Case, output: dict[str, Any], names: list[str] | None = None) -> GradeResult:
    """Run applicable graders and merge: score = mean of scores, passed = all passed, errors concatenated."""
    names = names or list(GRADERS)
    results = [r for n in names if (r := GRADERS[n](case, output)) is not None]
    if not results:
        return GradeResult("none", 0.0, False, [{"type": "no_applicable_grader", "detail": case.id}])
    return GradeResult(
        grader="+".join(r.grader for r in results),
        score=sum(r.score for r in results) / len(results),
        passed=all(r.passed for r in results),
        errors=[e for r in results for e in r.errors],
    )
