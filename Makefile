# ------------ basic settings (Linux / WSL) ------------
PYTHON_BIN ?= python3.11
VENV ?= .venv
VENV_BIN := $(VENV)/bin
PY  := $(VENV_BIN)/python
PIP := $(PY) -m pip
REQ ?= requirements.txt

# ------------ targets ---------------------------------
.PHONY: env dev eval-env lint test report anatomy-figure explorer clean

env:            ## create venv & install deps
	@echo "🔧  Creating virtual environment: $(VENV)"
	python3 -m venv $(VENV)
	@echo "⬆️   Upgrading pip + installing requirements"
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ)

dev:            ## create a lightweight CPU-only development environment
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[test]"

eval-env:       ## metrics/analysis environment (DINOv2, DreamSim, Orient Anything, pandas)
	$(PYTHON_BIN) -m venv .venv-eval
	.venv-eval/bin/python -m pip install --upgrade pip
	.venv-eval/bin/python -m pip install -r requirements-eval.txt

report:         ## rebuild results/study1 (tables, verdict, figures) from the committed metric CSVs
	.venv-eval/bin/python -m analysis.study1_report

anatomy-figure: ## redraw the LoRA subspace figure from results/lora_anatomy/summary.json
	.venv-eval/bin/python -m analysis.lora_anatomy --plot-only

explorer:       ## export thumbnails + data.json for docs/ (needs the generated images locally)
	.venv-eval/bin/python -m analysis.build_explorer

lint:           ## run ruff (check only)
	$(PY) -m ruff check .

test:           ## run pytest suite
	$(PY) -m pytest -q

clean:          ## remove caches & artefacts
	rm -rf .cache __pycache__ */__pycache__ *.egg-info $(VENV)
