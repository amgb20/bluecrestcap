.PHONY: ingest run test eval

VENV ?= ./myvenv
PYTHON ?= $(VENV)/bin/python
UVICORN ?= $(VENV)/bin/uvicorn
PYTEST ?= $(VENV)/bin/pytest

export PYTHONPATH := .

ingest:
	$(PYTHON) -m app.ingest

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTEST) -q

eval:
	$(PYTHON) eval/run_eval.py
