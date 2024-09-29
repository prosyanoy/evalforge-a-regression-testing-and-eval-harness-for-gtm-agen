"""Markdown run report and prompt-version trend chart."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evalforge.analysis import cluster_failures, render_clusters_md
from evalforge.dataset import TASKS, Case


def render_run_md(summary: dict[str, Any], cases: list[Case]) -> str:
    m = summary["metrics"]
    inputs = {c.id: c.input for c in cases}
    clusters = cluster_failures(summary["results"], inputs)
    backends = sorted({r["judge"]["backend"] for r in summary["results"]})
    lines = [f"# EvalForge report: `{summary['prompt_version']}`", "",
             f"Run `{summary['run_id']}`, {int(m['n_cases'])} cases, judge: {', '.join(backends)}", "",
             "| metric | value |", "|---|---|",
             f"| mean rule score | {m['mean_score']:.3f} |", f"| pass rate | {m['pass_rate']:.1%} |",
             f"| judge score (0-1) | {m['judge_score']:.3f} |", f"| judge pass rate | {m['judge_pass_rate']:.1%} |",
             f"| critical failures | {int(m['critical_failures'])} |",
             f"| latency p50 / p95 | {m['latency_p50_ms']:.2f} / {m['latency_p95_ms']:.2f} ms |",
             f"| tokens in / out | {int(m['tokens_in_total'])} / {int(m['tokens_out_total'])} |",
             f"| est. cost | ${m['cost_usd_total']:.5f} |", "",
             "| task | pass rate | mean score |", "|---|---|---|"]
    for t in TASKS:
        if f"pass_rate.{t}" in m:
            lines.append(f"| {t} | {m[f'pass_rate.{t}']:.1%} | {m[f'score.{t}']:.3f} |")
    lines += ["", render_clusters_md(clusters)]
    return "\n".join(lines)


def render_trend_png(runs: list[dict[str, Any]], path: Path) -> Path:
    """Line chart of quality metrics across prompt versions (latest run per version, in chronological order)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest: dict[str, dict[str, Any]] = {}
    for r in runs:  # runs are chronological; later runs replace earlier ones for the same version
        latest.pop(r["prompt_version"], None)
        latest[r["prompt_version"]] = r
    versions = list(latest)
    if not versions:
        raise ValueError("no runs to plot; run `evalforge run` or `evalforge demo` first")
    series = [("mean_score", "Mean rule score", "#2a6fdb"), ("pass_rate", "Case pass rate", "#1a9850"),
              ("judge_score", "Judge score", "#e08a00")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=130)
    for key, label, color in series:
        ys = [latest[v]["metrics"][key] for v in versions]
        ax.plot(versions, ys, marker="o", linewidth=2, color=color, label=f"{label} ({ys[-1]:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (0-1)")
    ax.set_xlabel("prompt version (chronological)")
    ax.set_title("EvalForge: quality across prompt versions")
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
