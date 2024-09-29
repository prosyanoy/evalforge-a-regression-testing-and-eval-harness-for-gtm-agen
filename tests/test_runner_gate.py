import json

import pytest

from evalforge import cli, gate
from evalforge.config import get_settings
from evalforge.graders.judge import RuleJudge
from evalforge.models import AgentResponse
from evalforge.runner import resolve_agent, run_eval
from evalforge.store import Store


def run(cases, version, **kw):
    return run_eval(resolve_agent("evalforge.agents.reference:build", version), cases, RuleJudge(), version, **kw)


def test_prompt_versions_improve_monotonically(cases):
    scores = [run(cases, v)["metrics"]["mean_score"] for v in ("v1", "v2", "v3")]
    assert scores[0] < scores[1] < scores[2] and scores[2] > 0.95


def test_metrics_and_cost(cases):
    m = run(cases, "v3")["metrics"]
    assert m["n_cases"] == 50 and m["tokens_in_total"] > 0 and m["tokens_out_total"] > 0
    expected = (m["tokens_in_total"] * 0.50 + m["tokens_out_total"] * 1.50) / 1e6
    assert m["cost_usd_total"] == pytest.approx(expected)
    assert m["latency_p95_ms"] >= m["latency_p50_ms"] >= 0
    assert {"pass_rate.summarize", "pass_rate.refusal", "score.triage"} <= set(m)


def test_agent_reported_usage_and_plain_callable(cases):
    def agent(req):
        return AgentResponse({"text": "hi", "action": "answer"}, input_tokens=1000, output_tokens=10)

    res = run_eval(agent, cases[:3], RuleJudge(), "custom")
    assert all(r["input_tokens"] == 1000 for r in res["results"])
    assert resolve_agent("tests.test_runner_gate:_plain", "x").__name__ == "_plain"


def _plain(req):
    return "ok"


def test_crashing_agent_is_a_failed_case_not_a_crash(cases):
    def agent(req):
        raise RuntimeError("boom")

    res = run_eval(agent, cases[:2], RuleJudge(), "bad")
    assert res["metrics"]["pass_rate"] == 0 and res["results"][0]["errors"][0]["type"] == "agent_exception"


def test_store_roundtrip(cases):
    summary = run(cases[:5], "v3")
    store = Store(":memory:")
    store.save_run(summary)
    got = store.get_run(summary["run_id"])
    assert got["metrics"] == summary["metrics"] and len(got["results"]) == 5
    assert store.latest_run_id() == summary["run_id"]


def test_gate_passes_on_same_and_fails_on_regression(cases):
    base = gate.make_baseline(run(cases, "v3"))
    same = gate.compare(run(cases, "v3"), base, get_settings())
    assert same["passed"] and not same["regressed_cases"]
    better_base = gate.make_baseline(run(cases, "v2"))
    assert gate.compare(run(cases, "v3"), better_base, get_settings())["passed"]  # improvement passes
    bad = gate.compare(run(cases, "v4-regressed"), base, get_settings())
    assert not bad["passed"] and bad["new_critical_failures"]
    assert any(not c["ok"] for c in bad["checks"] if c["name"] == "mean_score")
    assert "FAILED" in gate.render_gate_md(bad)


def test_gate_threshold_boundary(cases):
    base = gate.make_baseline(run(cases, "v3"))
    cur = run(cases, "v3")
    cur["metrics"]["mean_score"] = base["metrics"]["mean_score"] - 0.02
    assert gate.compare(cur, base, get_settings())["passed"]  # exactly at the limit is allowed
    cur["metrics"]["mean_score"] = base["metrics"]["mean_score"] - 0.021
    res = gate.compare(cur, base, get_settings())
    assert not res["passed"]
    assert gate.compare(cur, base, get_settings(), max_score_drop=0.05)["passed"]


def test_cli_run_then_gate_exit_codes(capsys, tmp_path):
    assert cli.main(["run", "--prompt-version", "v3", "--save-baseline", "--no-mlflow"]) == 0
    assert cli.main(["gate"]) == 0
    assert cli.main(["run", "--prompt-version", "v4-regressed", "--no-mlflow"]) == 0
    assert cli.main(["gate"]) == 1
    assert (get_settings().reports / "gate.md").read_text().startswith("## Regression gate: FAILED")
    assert cli.main(["gate", "--baseline", str(tmp_path / "missing.json")]) == 2


def test_cli_logs_to_mlflow(cases):
    import mlflow
    assert cli.main(["run", "--prompt-version", "v2", "--limit", "10"]) == 0
    mlflow.set_tracking_uri(get_settings().mlflow_uri)
    runs = mlflow.search_runs(experiment_names=["evalforge"])
    assert len(runs) == 1
    assert runs["metrics.mean_score"].iloc[0] > 0 and runs["tags.prompt_version"].iloc[0] == "v2"
    assert "metrics.cost_usd_total" in runs and "metrics.latency_p50_ms" in runs
