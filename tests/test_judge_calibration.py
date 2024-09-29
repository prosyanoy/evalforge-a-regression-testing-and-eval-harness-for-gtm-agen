import json

import pytest
from sklearn.metrics import cohen_kappa_score

from evalforge.agents.reference import build
from evalforge.config import get_settings
from evalforge.graders.calibration import calibration_report, cohens_kappa, interpret_kappa, render_calibration_md
from evalforge.graders.judge import FallbackJudge, OllamaJudge, RuleJudge, build_judge, parse_judge_reply
from evalforge.models import Request


def test_kappa_matches_sklearn():
    a = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
    b = [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]
    assert cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_kappa_edge_cases():
    assert cohens_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # degenerate but identical
    assert cohens_kappa([1, 1], [0, 0]) == pytest.approx(0.0, abs=1e-9) or cohens_kappa([1, 1], [0, 0]) <= 0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_interpretation_bands():
    assert interpret_kappa(0.9) == "almost perfect" and interpret_kappa(0.5) == "moderate" and interpret_kappa(-0.1) == "poor"


def test_rule_judge_separates_good_and_bad(cases):
    good, bad = build("v3"), build("v1")
    judge = RuleJudge()
    gp = bp = 0
    for c in cases:
        gp += judge.judge(c, good(Request(c.task, c.input))).passed
        bp += judge.judge(c, bad(Request(c.task, c.input))).passed
    assert gp >= 48 and bp <= 30


def test_rule_judge_is_deterministic(cases):
    c = cases[0]
    out = build("v2")(Request(c.task, c.input))
    assert RuleJudge().judge(c, out) == RuleJudge().judge(c, out)


GOOD_REPLY = json.dumps({"accuracy": 5, "completeness": 5, "safety": 5, "concision": 4, "rationale": "fine"})


def test_ollama_judge_parses_reply(cases):
    seen = {}

    def transport(url, payload, timeout):
        seen["url"] = url
        return {"message": {"content": "Sure! " + GOOD_REPLY}}

    v = OllamaJudge(get_settings(), transport).judge(cases[0], {"text": "x"})
    assert v.passed and v.overall == pytest.approx(4.75) and v.backend.startswith("ollama:")
    assert seen["url"].endswith("/api/chat")


def test_parse_rejects_malformed():
    with pytest.raises(ValueError):
        parse_judge_reply("no json here", "x")
    with pytest.raises(KeyError):
        parse_judge_reply('{"accuracy": 3}', "x")


def test_fallback_when_ollama_down(cases):
    def boom(url, payload, timeout):
        raise OSError("connection refused")

    primary = OllamaJudge(get_settings(), boom)
    assert not primary.available()
    v = FallbackJudge(primary).judge(cases[0], {"text": "x"})
    assert v.backend == "rule(fallback)"


def test_auto_judge_falls_back_to_rule_without_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    assert isinstance(build_judge("auto"), RuleJudge)
    with pytest.raises(ValueError):
        build_judge("bogus")


def test_calibration_report_perfect_and_flipped(cases):
    by_id = {c.id: c for c in cases}
    judge = RuleJudge()
    agent = build("v2")
    labels = []
    for c in cases:
        out = agent(Request(c.task, c.input))
        labels.append({"case_id": c.id, "output": out, "label": judge.judge(c, out).passed})
    rep = calibration_report(by_id, labels, judge)
    assert rep["kappa"] == 1.0 and rep["agreement"] == 1.0 and not rep["disagreements"]
    flipped = [{**l, "label": not l["label"]} if i < 10 else l for i, l in enumerate(labels)]
    rep2 = calibration_report(by_id, flipped, judge)
    assert rep2["agreement"] == pytest.approx(0.8) and len(rep2["disagreements"]) == 10 and rep2["kappa"] < 1
    assert "Cohen's kappa" in render_calibration_md(rep2)


def test_calibration_requires_labels(cases):
    with pytest.raises(ValueError):
        calibration_report({c.id: c for c in cases}, [], RuleJudge())
