"""Human review UI (FastAPI, no JS framework): label agent responses pass/fail and watch judge calibration."""
from __future__ import annotations

import html
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from evalforge.config import get_settings
from evalforge.dataset import load_cases
from evalforge.graders.calibration import calibration_report
from evalforge.graders.judge import build_judge
from evalforge.labels import merged_labels
from evalforge.models import response_key
from evalforge.store import Store

app = FastAPI(title="EvalForge review")


class LabelIn(BaseModel):
    run_id: str
    case_id: str
    label: bool
    reviewer: str = "human"
    note: str = ""


def _store() -> Store:
    return Store(get_settings().db)


def build_queue(store: Store, run_id: str | None = None) -> dict[str, Any]:
    """Review queue for a run: unlabelled first; within that, judge/rule disagreements and failures first."""
    run_id = run_id or store.latest_run_id()
    if not run_id:
        return {"run_id": None, "items": []}
    run = store.get_run(run_id)
    cases = {c.id: c for c in load_cases(get_settings().dataset)}
    labelled = {l["response_key"]: l["label"] for l in store.labels()}
    items = []
    for r in run["results"]:
        key = response_key(r["case_id"], r["output"])
        disagree = r["passed"] != r["judge"]["passed"]
        items.append({"case_id": r["case_id"], "task": r["task"], "severity": r["severity"],
                      "response_key": key, "input": cases[r["case_id"]].input if r["case_id"] in cases else "",
                      "expected": cases[r["case_id"]].expected if r["case_id"] in cases else {},
                      "output": r["output"], "rule_passed": r["passed"], "judge_passed": r["judge"]["passed"],
                      "judge_rationale": r["judge"]["rationale"], "errors": r["errors"], "disagree": disagree,
                      "label": labelled.get(key)})
    items.sort(key=lambda i: (i["label"] is not None, not i["disagree"], i["rule_passed"], -i["severity"], i["case_id"]))
    return {"run_id": run_id, "prompt_version": run["prompt_version"], "items": items}


@app.get("/api/queue")
def api_queue(run_id: str | None = None) -> dict[str, Any]:
    store = _store()
    try:
        return build_queue(store, run_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    finally:
        store.close()


@app.post("/api/labels")
def api_label(body: LabelIn) -> dict[str, Any]:
    store = _store()
    try:
        run = store.get_run(body.run_id)
        row = next((r for r in run["results"] if r["case_id"] == body.case_id), None)
        if row is None:
            raise HTTPException(404, f"case {body.case_id} not in run {body.run_id}")
        key = response_key(body.case_id, row["output"])
        store.add_label(key, body.case_id, row["output"], body.label, body.reviewer, body.note)
        return {"ok": True, "response_key": key}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    finally:
        store.close()


@app.get("/api/calibration")
def api_calibration(judge: str = "rule") -> dict[str, Any]:
    s = get_settings()
    store = _store()
    try:
        cases = {c.id: c for c in load_cases(s.dataset)}
        labels = merged_labels(s.human_labels, store)
        try:
            return calibration_report(cases, labels, build_judge(judge))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    finally:
        store.close()


PAGE = """<!doctype html><meta charset=utf-8><title>EvalForge review</title>
<style>
body{font:14px system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1b1f24}
.card{border:1px solid #d0d7de;border-radius:8px;padding:12px 16px;margin:12px 0}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;background:#eef1f4;margin-right:4px}
.ok{background:#dafbe1}.bad{background:#ffebe9}.warn{background:#fff8c5}
pre{white-space:pre-wrap;background:#f6f8fa;padding:8px;border-radius:6px;margin:6px 0}
button{padding:4px 12px;margin-right:6px;cursor:pointer}
</style>
<h1>EvalForge: human review</h1>
<p>Run <code>__RUN__</code> (<b>__VERSION__</b>). Label each response pass/fail; labels feed the
<a href="/api/calibration">judge calibration report</a>. Disagreements between rule grader and judge come first.</p>
<label>Reviewer <input id=rv value="human" size=12></label>
__CARDS__
<script>
async function label(runId, caseId, val){
  const r = await fetch('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({run_id:runId,case_id:caseId,label:val,reviewer:document.getElementById('rv').value})});
  if(r.ok){document.getElementById('s-'+caseId).textContent='saved: '+(val?'PASS':'FAIL');}
  else{document.getElementById('s-'+caseId).textContent='error '+r.status;}
}
</script>"""


def _pill(ok: bool, text: str) -> str:
    return f'<span class="tag {"ok" if ok else "bad"}">{html.escape(text)}</span>'


@app.get("/", response_class=HTMLResponse)
def index(run_id: str | None = None) -> str:
    store = _store()
    try:
        q = build_queue(store, run_id)
    except KeyError:
        raise HTTPException(404, "unknown run") from None
    finally:
        store.close()
    if not q["run_id"]:
        return "<p>No runs yet. Run <code>python -m evalforge run</code> first.</p>"
    cards = []
    for it in q["items"]:
        status = {None: "unlabelled", True: "labelled PASS", False: "labelled FAIL"}[it["label"]]
        cid = html.escape(it["case_id"])
        cards.append(
            f'<div class=card><b>{cid}</b> <span class=tag>{html.escape(it["task"])}</span>'
            f'<span class=tag>severity {it["severity"]}</span>{_pill(it["rule_passed"], "rule " + ("pass" if it["rule_passed"] else "fail"))}'
            f'{_pill(it["judge_passed"], "judge " + ("pass" if it["judge_passed"] else "fail"))}'
            f'{"<span class='tag warn'>disagreement</span>" if it["disagree"] else ""}'
            f'<details><summary>input</summary><pre>{html.escape(it["input"])}</pre></details>'
            f'<div>Agent output</div><pre>{html.escape(json.dumps(it["output"], indent=2))}</pre>'
            f'<div>Expected</div><pre>{html.escape(json.dumps(it["expected"]))}</pre>'
            f'<div>Judge: {html.escape(it["judge_rationale"])}</div>'
            f'<p><button onclick="label(\'{html.escape(q["run_id"])}\',\'{cid}\',true)">Pass</button>'
            f'<button onclick="label(\'{html.escape(q["run_id"])}\',\'{cid}\',false)">Fail</button>'
            f'<span id="s-{cid}">{status}</span></p></div>')
    return (PAGE.replace("__RUN__", html.escape(q["run_id"])).replace("__VERSION__", html.escape(q["prompt_version"]))
            .replace("__CARDS__", "\n".join(cards)))
