"""Deterministic reference GTM agent with several "prompt versions".

It stands in for an LLM-backed agent so the whole harness runs offline and reproducibly. Each version is a
feature set; later versions fix failure modes of earlier ones (and `v4-regressed` deliberately breaks things so
the regression gate has something to catch). Plug in your own agent with `--agent pkg.module:callable`.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from evalforge.models import Request

VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {"summary": "first2", "steps": "basic", "triage": "basic", "priority": "flat", "refusal": "none"},
    "v2": {"summary": "top3", "steps": "full", "triage": "keyword", "priority": "urgent", "refusal": "basic"},
    "v3": {"summary": "signal", "steps": "hedge", "triage": "full", "priority": "rules", "refusal": "full"},
    "v4-regressed": {"summary": "top3", "steps": "full", "triage": "keyword", "priority": "flat", "refusal": "basic"},
}

_TURN = re.compile(r"^(Rep|Prospect):\s*(.*)$")
_SENT = re.compile(r"(?<=[.!?])\s+")
_SIGNAL = re.compile(r"\b(budget|go live|evaluating|biggest pain|signs off)\b|\$\d", re.I)
_COMMIT_BASIC = re.compile(r"\b(I'll|I will)\b", re.I)
_COMMIT_FULL = re.compile(r"\b(I'll|I will|we'll|we will|let me|I can)\b", re.I)
_HEDGE = re.compile(r"\b(maybe|might|perhaps|possibly|no commitment|could)\b", re.I)


def _sentences(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        m = _TURN.match(line.strip())
        speaker, body = (m.group(1), m.group(2)) if m else ("Rep", line.strip())
        out += [(speaker, s) for s in _SENT.split(body) if s]
    return out


def summarize(text: str, mode: str) -> dict[str, Any]:
    sents = [s for _, s in _sentences(text)]
    if mode == "first2":
        chosen = sents[:2]
    else:
        signal = [s for s in sents if _SIGNAL.search(s)]
        chosen = signal[:3] if mode == "top3" else signal[:6]
    return {"text": " ".join(chosen)}


def next_steps(text: str, mode: str) -> dict[str, Any]:
    pattern = _COMMIT_BASIC if mode == "basic" else _COMMIT_FULL
    steps = []
    for speaker, sent in _sentences(text):
        if not pattern.search(sent):
            continue
        if mode == "hedge" and _HEDGE.search(sent):
            continue
        owner = "rep" if mode == "basic" or speaker == "Rep" else "prospect"
        steps.append({"owner": owner, "task": sent})
    return {"text": "; ".join(s["task"] for s in steps), "next_steps": steps}


_CATS_BASIC = [("billing", r"invoice|charge|refund|billing"), ("bug", r"error|crash|broken|not working|\b500\b"),
               ("account_access", r"password|log in|login|locked out")]
_CATS_KEYWORD = _CATS_BASIC + [
    ("how_to", r"how do i|how can i"), ("feature_request", r"would be great|feature|wish"),
    ("security", r"breach|vulnerab|phishing|leaked")]
_SECURITY = ("security", r"breach|vulnerab|phishing|leaked")
_CATS_FULL = [_SECURITY] + [c for c in _CATS_KEYWORD if c[0] != "security"]  # security is checked first


def triage(text: str, cat_mode: str, prio_mode: str) -> dict[str, Any]:
    table = {"basic": _CATS_BASIC, "keyword": _CATS_KEYWORD, "full": _CATS_FULL}[cat_mode]
    low = text.lower()
    category = next((name for name, pat in table if re.search(pat, low)), "other")
    if prio_mode == "rules":
        if re.search(r"\bdown\b|all users|leaked|breach|outage", low):
            priority = "P1"
        elif category in ("how_to", "feature_request"):
            priority = "P3"
        else:
            priority = "P2"
    elif prio_mode == "urgent":
        priority = "P1" if re.search(r"urgent|asap|\bdown\b", low) else "P2"
    else:
        priority = "P2"
    return {"text": f"Category: {category}. Priority: {priority}.", "category": category, "priority": priority}


_REFUSE_BASIC = r"testimonial from|personal cell|unsubscribed|opt-out|another customer"
_REFUSE_FULL = (r"testimonial from|personal cell|(unsubscribed|opt-out).*(anyway|ignore)|"
                r"another customer|not yet but|never said|she never")
_ESCALATE = r"\bsue\b|attorney|lawsuit|legal action|\d+% discount|breach"


def refusal(text: str, mode: str) -> dict[str, Any]:
    low = text.lower()
    if mode != "none" and re.search(_REFUSE_BASIC if mode == "basic" else _REFUSE_FULL, low):
        return {"action": "refuse", "text": "I can't help with that: it is deceptive or violates privacy/consent policy. "
                "I can help with a compliant alternative instead."}
    if mode == "full" and re.search(_ESCALATE, low):
        return {"action": "escalate", "text": "This needs human review. I'm routing it to a manager and legal."}
    return {"action": "answer", "text": "Sure. Here is a short draft you can send: Hi, would you have 20 minutes next week to talk?"}


def build(prompt_version: str = "v3") -> Callable[[Request], dict[str, Any]]:
    """Factory: returns the agent callable for a prompt version."""
    if prompt_version not in VERSIONS:
        raise KeyError(f"unknown prompt version '{prompt_version}'; known: {sorted(VERSIONS)}")
    cfg = VERSIONS[prompt_version]

    def agent(req: Request) -> dict[str, Any]:
        if req.task == "summarize":
            return summarize(req.text, cfg["summary"])
        if req.task == "next_steps":
            return next_steps(req.text, cfg["steps"])
        if req.task == "triage":
            return triage(req.text, cfg["triage"], cfg["priority"])
        if req.task == "refusal":
            return refusal(req.text, cfg["refusal"])
        raise ValueError(f"unsupported task {req.task}")

    agent.__name__ = f"reference_{prompt_version}"
    return agent
