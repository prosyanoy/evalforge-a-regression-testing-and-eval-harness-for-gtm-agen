"""Failure analysis: cluster failed cases by error type + transcript content, rank by frequency x severity."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

FIX_HINTS = {
    "unsafe_compliance": "Add explicit refusal rules/examples for deceptive, privacy-violating or consent-violating requests.",
    "missed_escalation": "Add escalation triggers (legal threats, breaches, out-of-policy discounts) and route to a human.",
    "over_refusal": "Narrow refusal rules; add benign counter-examples (e.g. compliant opt-out footers).",
    "missing_fact": "Raise extraction recall: include budget, timeline, competitor, pain and decision-maker signals.",
    "forbidden_content": "Filter small talk / irrelevant content out of summaries.",
    "too_long": "Enforce a summary length budget.",
    "missed_action": "Capture every commitment ('we will', 'let me', 'I can'), not only first-person 'I'll'.",
    "wrong_owner": "Assign the action owner from the speaker of the commitment, not a default.",
    "spurious_action": "Filter hedged / non-committal statements ('maybe', 'might', 'no commitment').",
    "wrong_category": "Extend category keywords; check precedence (e.g. security before bug).",
    "wrong_priority": "Add priority rules for outage / blocked-team / security signals.",
    "agent_exception": "Fix the crash: the agent raised an exception.",
}


def _case_text(r: dict[str, Any], inputs: dict[str, str]) -> str:
    types = " ".join(e["type"] for e in r["errors"])
    details = " ".join(e["detail"] for e in r["errors"])
    # error types are repeated so they dominate the TF-IDF space: clusters group by error type first
    return f"{r['task']} {(types + ' ') * 3}{details} {inputs.get(r['case_id'], '')}"


def _choose_k(X: Any, n: int) -> int:
    best_k, best = 1, -1.0
    for k in range(2, min(6, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sc = silhouette_score(X, labels)
        if sc > best:
            best_k, best = k, sc
    return best_k


def cluster_failures(results: list[dict[str, Any]], inputs: dict[str, str]) -> list[dict[str, Any]]:
    """Return clusters sorted by priority = size x mean severity (highest first)."""
    failed = [r for r in results if not r["passed"]]
    if not failed:
        return []
    texts = [_case_text(r, inputs) for r in failed]
    if len(failed) < 4:
        groups = list(range(len(failed)))
        labels = np.array(groups)
        X = None
    else:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(texts)
        k = _choose_k(X, len(failed))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X) if k > 1 else np.zeros(len(failed), int)
        terms = np.array(vec.get_feature_names_out())
    clusters = []
    for cid in sorted(set(labels.tolist())):
        idx = [i for i, l in enumerate(labels.tolist()) if l == cid]
        members = [failed[i] for i in idx]
        types = Counter(e["type"] for m in members for e in m["errors"])
        top_terms: list[str] = []
        if X is not None:
            centroid = np.asarray(X[idx].mean(axis=0)).ravel()
            top_terms = [t for t in terms[np.argsort(centroid)[::-1][:12]] if t not in types][:4]
        mean_sev = sum(m["severity"] for m in members) / len(members)
        dominant = types.most_common(1)[0][0] if types else "unknown"
        rep = min(members, key=lambda m: m["score"])
        clusters.append({
            "label": dominant, "error_types": dict(types.most_common()), "size": len(members),
            "mean_severity": round(mean_sev, 2), "priority": round(len(members) * mean_sev, 2),
            "tasks": sorted({m["task"] for m in members}), "top_terms": top_terms,
            "case_ids": sorted(m["case_id"] for m in members),
            "example": {"case_id": rep["case_id"], "error": rep["errors"][0]["detail"] if rep["errors"] else ""},
            "fix": FIX_HINTS.get(dominant, "Inspect the failing cases."),
        })
    clusters.sort(key=lambda c: (-c["priority"], -c["size"], c["label"]))
    for rank, c in enumerate(clusters, 1):
        c["rank"] = rank
    return clusters


def render_clusters_md(clusters: list[dict[str, Any]]) -> str:
    if not clusters:
        return "## Failure analysis\n\nNo failures. Nothing to fix.\n"
    lines = ["## Failure analysis: what to fix first", "",
             "| # | cluster (dominant error) | cases | mean severity | priority | tasks | fix |",
             "|---|---|---|---|---|---|---|"]
    for c in clusters:
        lines.append(f"| {c['rank']} | `{c['label']}` | {c['size']} | {c['mean_severity']} | {c['priority']} | "
                     f"{', '.join(c['tasks'])} | {c['fix']} |")
    lines.append("")
    for c in clusters:
        lines.append(f"- **#{c['rank']} `{c['label']}`** ({c['size']} cases: {', '.join(c['case_ids'][:8])}"
                     f"{'...' if c['size'] > 8 else ''}); example {c['example']['case_id']}: {c['example']['error']}")
    return "\n".join(lines) + "\n"
