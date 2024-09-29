from evalforge.dataset import Case
from evalforge.graders import GRADERS, grade_case, register_grader
from evalforge.models import GradeResult


def mk(task, expected, text="x"):
    return Case("c1", task, text, expected, 1)


def test_exact_grader_names_refusal_errors():
    case = mk("refusal", {"fields": {"action": "refuse"}})
    r = grade_case(case, {"action": "answer", "text": ""})
    assert not r.passed and r.errors[0]["type"] == "unsafe_compliance"
    esc = grade_case(mk("refusal", {"fields": {"action": "escalate"}}), {"action": "answer"})
    assert esc.errors[0]["type"] == "missed_escalation"
    over = grade_case(mk("refusal", {"fields": {"action": "answer"}}), {"action": "refuse"})
    assert over.errors[0]["type"] == "over_refusal"
    assert grade_case(case, {"action": "REFUSE"}).passed


def test_exact_partial_credit():
    r = grade_case(mk("triage", {"fields": {"category": "bug", "priority": "P1"}}), {"category": "bug", "priority": "P2"})
    assert r.score == 0.5 and r.errors[0]["type"] == "wrong_priority"


def test_keyword_grader():
    case = mk("summarize", {"must_include": ["$40k", "March"], "must_not_include": ["golf"], "max_words": 5})
    ok = grade_case(case, {"text": "budget $40k by March"})
    assert ok.passed and ok.score == 1.0
    bad = grade_case(case, {"text": "we talked golf and a lot about many other things"})
    kinds = {e["type"] for e in bad.errors}
    assert kinds == {"missing_fact", "forbidden_content", "too_long"}


def test_actions_grader_f1_and_error_types():
    case = mk("next_steps", {"actions": [{"owner": "rep", "keywords": ["pricing"]},
                                          {"owner": "prospect", "keywords": ["contract"]}],
                             "not_actions": ["revisit"]})
    perfect = {"next_steps": [{"owner": "rep", "task": "Send pricing deck"}, {"owner": "prospect", "task": "Share contract"}]}
    assert grade_case(case, perfect).passed
    wrong_owner = {"next_steps": [{"owner": "rep", "task": "Send pricing deck"}, {"owner": "rep", "task": "Share contract"}]}
    r = grade_case(case, wrong_owner)
    assert [e["type"] for e in r.errors] == ["wrong_owner"] and not r.passed
    missed = grade_case(case, {"next_steps": [{"owner": "rep", "task": "Send pricing deck"},
                                              {"owner": "prospect", "task": "revisit later"}]})
    assert {e["type"] for e in missed.errors} == {"missed_action", "spurious_action"}
    assert 0 < missed.score < 1


def test_custom_grader_registration():
    @register_grader("always_fail")
    def _g(case, output):
        return GradeResult("always_fail", 0.0, False, [{"type": "custom", "detail": "no"}])
    try:
        r = grade_case(mk("triage", {"fields": {"a": "b"}}), {"a": "b"}, names=["exact", "always_fail"])
        assert not r.passed and r.score == 0.5
    finally:
        GRADERS.pop("always_fail")
