PY ?= .venv/bin/python

.PHONY: setup test demo run gate review mlflow-ui calibrate
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
test:
	$(PY) -m pytest -q
demo:        ## run v1..v3 and draw reports/trend.png
	$(PY) -m evalforge demo
run:         ## evaluate the current agent (PROMPT_VERSION=v3) and write reports/current.json
	$(PY) -m evalforge run
gate: run    ## fail (exit 1) if quality regressed vs baselines/baseline.json
	$(PY) -m evalforge gate
calibrate:
	$(PY) -m evalforge calibrate
review:
	$(PY) -m evalforge review
mlflow-ui:
	$(PY) -m mlflow server --backend-store-uri sqlite:///data/mlflow.db --port 5000
