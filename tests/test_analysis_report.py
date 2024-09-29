from evalforge.analysis import cluster_failures, render_clusters_md
from evalforge.graders.judge import RuleJudge
from evalforge.report import render_run_md, render_trend_png
from evalforge.runner import resolve_agent, run_eval


def run(cases, v):
    return run_eval(resolve_agent("evalforge.agents.reference:build", v), cases, RuleJudge(), v)


def test_clusters_ranked_by_frequency_and_severity(cases):
    res = run(cases, "v1")
    clusters = cluster_failures(res["results"], {c.id: c.input for c in cases})
    assert clusters
    assert sum(c["size"] for c in clusters) == sum(not r["passed"] for r in res["results"])
    priorities = [c["priority"] for c in clusters]
    assert priorities == sorted(priorities, reverse=True)
    assert [c["rank"] for c in clusters] == list(range(1, len(clusters) + 1))
    # v1 never refuses: unsafe_compliance must show up as a (high severity) cluster
    assert any("unsafe_compliance" in c["error_types"] for c in clusters)
    assert all(c["fix"] for c in clusters)
    assert "what to fix first" in render_clusters_md(clusters)


def test_clustering_is_deterministic(cases):
    res = run(cases, "v2")
    inputs = {c.id: c.input for c in cases}
    assert cluster_failures(res["results"], inputs) == cluster_failures(res["results"], inputs)


def test_clusters_edge_cases(cases):
    inputs = {c.id: c.input for c in cases}
    assert cluster_failures([], inputs) == []
    assert "No failures" in render_clusters_md([])
    few = [r for r in run(cases, "v1")["results"] if not r["passed"]][:2]
    assert sum(c["size"] for c in cluster_failures(few, inputs)) == 2


def test_report_and_trend_chart(cases, tmp_path):
    runs = [run(cases, v) for v in ("v1", "v2", "v3")]
    md = render_run_md(runs[0], cases)
    assert "EvalForge report" in md and "Failure analysis" in md
    png = render_trend_png(runs, tmp_path / "t.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
