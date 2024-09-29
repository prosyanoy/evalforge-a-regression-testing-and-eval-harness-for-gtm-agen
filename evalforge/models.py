"""Core data types shared across the harness."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Request:
    """What an agent receives: the task type and the raw text (transcript, ticket, message)."""
    task: str
    text: str


@dataclass
class AgentResponse:
    """Normalised agent output. `output` is a dict, e.g. {"text": ..., "action": ..., "next_steps": [...]}."""
    output: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class GradeResult:
    grader: str
    score: float  # 0..1
    passed: bool
    errors: list[dict[str, str]] = field(default_factory=list)  # [{"type": ..., "detail": ...}]


@dataclass
class Verdict:
    scores: dict[str, float]  # rubric criterion -> 1..5
    overall: float  # mean of criteria, 1..5
    passed: bool
    rationale: str
    backend: str  # "rule", "ollama:<model>", "rule(fallback)"


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token) used when the agent reports no usage."""
    return max(1, round(len(text) / 4))


def normalize_response(raw: Any, request: Request) -> AgentResponse:
    """Accept str | dict | AgentResponse from any agent function."""
    if isinstance(raw, AgentResponse):
        resp = raw
    elif isinstance(raw, str):
        resp = AgentResponse(output={"text": raw})
    elif isinstance(raw, dict):
        resp = AgentResponse(output=dict(raw))
    else:
        raise TypeError(f"agent must return str, dict or AgentResponse, got {type(raw).__name__}")
    out_text = json.dumps(resp.output, sort_keys=True)
    if resp.input_tokens is None:
        resp.input_tokens = estimate_tokens(request.text)
    if resp.output_tokens is None:
        resp.output_tokens = estimate_tokens(out_text)
    return resp


def response_key(case_id: str, output: dict[str, Any]) -> str:
    """Stable identity of (case, agent output); links human labels to the exact response they judged."""
    blob = json.dumps({"case": case_id, "output": output}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]
