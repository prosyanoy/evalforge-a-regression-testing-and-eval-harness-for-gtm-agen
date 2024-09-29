"""Judge calibration: agreement between the judge and human labels (Cohen's kappa)."""
from __future__ import annotations

import random
from typing import Any, Sequence

from evalforge.dataset import Case
from evalforge.graders.base import grade_case
from evalforge.graders.judge import Judge


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Unweighted Cohen's kappa for two raters over the same items. Degenerate (pe == 1) -> 1.0 if identical else 0.0."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label sequences")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in set(a) | set(b))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) bands."""
    for limit, label in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= limit:
            return label
    return "almost perfect"


def _bootstrap_ci(a: list[bool], b: list[bool], iters: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    ks = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        ks.append(cohens_kappa([a[i] for i in idx], [b[i] for i in idx]))
    ks.sort()
    return ks[int(0.025 * iters)], ks[int(0.975 * iters) - 1]


def _confusion(judge: list[bool], human: list[bool]) -> dict[str, int]:
    return {"both_pass": sum(j and h for j, h in zip(judge, human)),
            "both_fail": sum((not j) and (not h) for j, h in zip(judge, human)),
            "judge_pass_human_fail": sum(j and not h for j, h in zip(judge, human)),
            "judge_fail_human_pass": sum((not j) and h for j, h in zip(judge, human))}


def calibration_report(cases: dict[str, Case], labels: list[dict[str, Any]], judge: Judge) -> dict[str, Any]:
    """labels: [{case_id, output, label(bool pass), reviewer}]. Re-judges each labelled response."""
    rows = [l for l in labels if l["case_id"] in cases]
    if not rows:
        raise ValueError("no human labels match cases in the dataset")
    human = [bool(r["label"]) for r in rows]
    verdicts = [judge.judge(cases[r["case_id"]], r["output"]) for r in rows]
    jpass = [v.passed for v in verdicts]
    rpass = [grade_case(cases[r["case_id"]], r["output"]).passed for r in rows]
    kappa = cohens_kappa(jpass, human)
    lo, hi = _bootstrap_ci(jpass, human)
    warnings = []
    if len(rows) < 20:
        warnings.append(f"only {len(rows)} labels; kappa is unstable below ~20")
    if len(set(human)) < 2:
        warnings.append("human labels contain a single class; kappa is uninformative")
    disagreements = [{"case_id": r["case_id"], "human": h, "judge": j, "judge_rationale": v.rationale,
                      "judge_overall": v.overall}
                     for r, h, j, v in zip(rows, human, jpass, verdicts) if h != j]
    return {
        "n_labels": len(rows),
        "judge_backend": sorted({v.backend for v in verdicts}),
        "agreement": sum(x == y for x, y in zip(jpass, human)) / len(rows),
        "kappa": kappa, "kappa_ci95": [lo, hi], "interpretation": interpret_kappa(kappa),
        "confusion": _confusion(jpass, human),
        "rule_grader_kappa": cohens_kappa(rpass, human),
        "human_pass_rate": sum(human) / len(human), "judge_pass_rate": sum(jpass) / len(jpass),
        "disagreements": disagreements, "warnings": warnings,
    }


def render_calibration_md(rep: dict[str, Any]) -> str:
    c = rep["confusion"]
    lines = ["# Judge calibration report", "",
             f"- Labelled responses: **{rep['n_labels']}**  (judge backend: {', '.join(rep['judge_backend'])})",
             f"- Raw agreement: **{rep['agreement']:.1%}**",
             f"- Cohen's kappa: **{rep['kappa']:.3f}** ({rep['interpretation']}), 95% bootstrap CI "
             f"[{rep['kappa_ci95'][0]:.2f}, {rep['kappa_ci95'][1]:.2f}]",
             f"- Reference: rule-grader vs human kappa = {rep['rule_grader_kappa']:.3f}",
             f"- Pass rates: human {rep['human_pass_rate']:.1%}, judge {rep['judge_pass_rate']:.1%}", "",
             "| | human pass | human fail |", "|---|---|---|",
             f"| judge pass | {c['both_pass']} | {c['judge_pass_human_fail']} |",
             f"| judge fail | {c['judge_fail_human_pass']} | {c['both_fail']} |", ""]
    for w in rep["warnings"]:
        lines.append(f"> warning: {w}")
    if rep["disagreements"]:
        lines += ["", "## Disagreements", "", "| case | human | judge | judge rationale |", "|---|---|---|---|"]
        for d in rep["disagreements"]:
            lines.append(f"| {d['case_id']} | {'pass' if d['human'] else 'fail'} | {'pass' if d['judge'] else 'fail'} "
                         f"| {d['judge_rationale']} |")
    verdict = ("trustworthy enough to gate on" if rep["kappa"] >= 0.6 else
               "NOT yet reliable: add labels or fix the rubric before trusting judge scores")
    lines += ["", f"**Verdict:** judge is {verdict}."]
    return "\n".join(lines) + "\n"
