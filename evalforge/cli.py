"""`evalforge` command line: run | gate | analyze | calibrate | trend | demo | review | seed-labels."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from evalforge import gate as gate_mod
from evalforge import tracking
from evalforge.analysis import cluster_failures, render_clusters_md
from evalforge.config import get_settings
from evalforge.dataset import DatasetError, load_cases
from evalforge.graders.calibration import calibration_report, render_calibration_md
from evalforge.graders.judge import build_judge
from evalforge.labels import merged_labels
from evalforge.report import render_run_md, render_trend_png
from evalforge.runner import DEFAULT_AGENT, resolve_agent, run_eval
from evalforge.store import Store


def _execute(args: argparse.Namespace, version: str, judge_kind: str | None = None, quiet: bool = False) -> dict:
    s = get_settings()
    cases = load_cases(args.dataset or s.dataset)
    if getattr(args, "limit", None):
        cases = cases[: args.limit]
    agent_spec = getattr(args, "agent", None) or DEFAULT_AGENT
    judge = build_judge(judge_kind or getattr(args, "judge", None))
    summary = run_eval(resolve_agent(agent_spec, version), cases, judge, version, agent_name=agent_spec, settings=s)
    md = render_run_md(summary, cases)
    artifacts = {"report.md": md}
    if not getattr(args, "no_mlflow", False):
        summary["mlflow_run_id"] = tracking.log_run(summary, artifacts, s)
    store = Store(s.db)
    store.save_run(summary)
    store.close()
    out_dir = s.reports / summary["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(md)
    compact = {k: v for k, v in summary.items() if k != "results"} | {
        "results": [{k: v for k, v in r.items() if k != "input"} for r in summary["results"]]}
    (s.reports / "current.json").write_text(json.dumps(compact, indent=2))
    (s.reports / "latest.md").write_text(md)
    if not quiet:
        print(md)
        print(f"report: {out_dir / 'report.md'}   mlflow run: {summary.get('mlflow_run_id')}")
    return summary


def cmd_run(args: argparse.Namespace) -> int:
    s = get_settings()
    s.reports.mkdir(parents=True, exist_ok=True)
    summary = _execute(args, args.prompt_version)
    if args.save_baseline:
        gate_mod.save_baseline(summary, s.baseline)
        print(f"baseline written to {s.baseline}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    s = get_settings()
    cur_path = Path(args.current) if args.current else s.reports / "current.json"
    base_path = Path(args.baseline) if args.baseline else s.baseline
    if not cur_path.is_file() or not base_path.is_file():
        print(f"gate error: need both current run ({cur_path}) and baseline ({base_path})", file=sys.stderr)
        return 2
    result = gate_mod.compare(json.loads(cur_path.read_text()), json.loads(base_path.read_text()), s, args.max_drop)
    md = gate_mod.render_gate_md(result)
    print(md)
    (s.reports / "gate.md").parent.mkdir(parents=True, exist_ok=True)
    (s.reports / "gate.md").write_text(md)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as fh:
            fh.write(md)
    return 0 if result["passed"] else 1


def _load_run(args: argparse.Namespace):
    s = get_settings()
    store = Store(s.db)
    run_id = args.run_id or store.latest_run_id()
    if not run_id:
        raise SystemExit("no runs found; run `evalforge run` first")
    return s, store, store.get_run(run_id)


def cmd_analyze(args: argparse.Namespace) -> int:
    s, store, run = _load_run(args)
    inputs = {c.id: c.input for c in load_cases(s.dataset)}
    clusters = cluster_failures(run["results"], inputs)
    print(render_clusters_md(clusters))
    if args.json:
        print(json.dumps(clusters, indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    s = get_settings()
    store = Store(s.db)
    cases = {c.id: c for c in load_cases(s.dataset)}
    labels = merged_labels(s.human_labels, store)
    rep = calibration_report(cases, labels, build_judge(args.judge))
    md = render_calibration_md(rep)
    s.reports.mkdir(parents=True, exist_ok=True)
    (s.reports / "calibration.md").write_text(md)
    (s.reports / "calibration.json").write_text(json.dumps(rep, indent=2))
    print(md)
    return 0


def cmd_trend(args: argparse.Namespace) -> int:
    s = get_settings()
    path = render_trend_png(Store(s.db).list_runs(), Path(args.out) if args.out else s.reports / "trend.png")
    print(f"trend chart written to {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run every reference prompt version in order (v1 -> v3), then draw the trend chart."""
    s = get_settings()
    s.reports.mkdir(parents=True, exist_ok=True)
    args.agent = None
    for v in ("v1", "v2", "v3"):
        summary = _execute(args, v, quiet=True)
        m = summary["metrics"]
        print(f"{v}: mean_score={m['mean_score']:.3f} pass_rate={m['pass_rate']:.1%} judge={m['judge_score']:.3f}")
    path = render_trend_png(Store(s.db).list_runs(), s.reports / "trend.png")
    print(f"trend chart: {path}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("evalforge.review.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    from evalforge.seed import write_seed_labels
    s = get_settings()
    n = write_seed_labels(load_cases(s.dataset), s.human_labels)
    print(f"wrote {n} synthetic human labels to {s.human_labels}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an agent against the golden dataset and log to MLflow + SQLite")
    r.add_argument("--agent", help=f"module:callable (default {DEFAULT_AGENT})")
    r.add_argument("--prompt-version", default=os.environ.get("PROMPT_VERSION", "v3"))
    r.add_argument("--dataset")
    r.add_argument("--judge", choices=["auto", "rule", "ollama"])
    r.add_argument("--limit", type=int)
    r.add_argument("--no-mlflow", action="store_true")
    r.add_argument("--save-baseline", action="store_true", help="write this run as the committed baseline")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("gate", help="compare reports/current.json to the baseline; exit 1 on regression")
    g.add_argument("--current")
    g.add_argument("--baseline")
    g.add_argument("--max-drop", type=float, help="override max allowed mean_score drop")
    g.set_defaults(fn=cmd_gate)

    a = sub.add_parser("analyze", help="cluster failures of a run and rank what to fix first")
    a.add_argument("--run-id")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    c = sub.add_parser("calibrate", help="judge vs human agreement (Cohen's kappa)")
    c.add_argument("--judge", choices=["auto", "rule", "ollama"])
    c.set_defaults(fn=cmd_calibrate)

    t = sub.add_parser("trend", help="draw the quality-by-prompt-version trend chart")
    t.add_argument("--out")
    t.set_defaults(fn=cmd_trend)

    d = sub.add_parser("demo", help="run reference prompt versions v1..v3 and draw the trend chart")
    d.add_argument("--dataset")
    d.add_argument("--judge", choices=["auto", "rule", "ollama"], default="rule")
    d.add_argument("--no-mlflow", action="store_true")
    d.set_defaults(fn=cmd_demo)

    v = sub.add_parser("review", help="serve the human review UI")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(fn=cmd_review)

    sd = sub.add_parser("seed-labels", help="regenerate the synthetic human-label subset")
    sd.set_defaults(fn=cmd_seed)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (DatasetError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
