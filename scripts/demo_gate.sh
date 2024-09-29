#!/usr/bin/env bash
# Reproduces the "PR fails the gate, then passes after the fix" story locally (record this for a GIF/asciicast).
# Uses throwaway state under a temp dir; the committed baseline is only read.
set -u
PY=${PY:-.venv/bin/python}
TMP=$(mktemp -d)
export EVALFORGE_DB=$TMP/ef.db EVALFORGE_REPORTS=$TMP/reports MLFLOW_TRACKING_URI=sqlite:///$TMP/mlflow.db EVALFORGE_JUDGE=rule
echo "### PR #1: prompt change (v4-regressed) -> gate"
$PY -m evalforge run --prompt-version v4-regressed --no-mlflow >/dev/null 2>&1
$PY -m evalforge gate; echo "gate exit code: $?  (build FAILS)"
echo; echo "### PR #2: fix pushed (v3) -> gate"
$PY -m evalforge run --prompt-version v3 --no-mlflow >/dev/null 2>&1
$PY -m evalforge gate; echo "gate exit code: $?  (build PASSES)"
rm -rf "$TMP"
