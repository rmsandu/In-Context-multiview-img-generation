# ------------ basic settings (Linux / WSL) ------------
PYTHON_BIN ?= python3.11
VENV ?= .venv
VENV_BIN := $(VENV)/bin
PY  := $(VENV_BIN)/python
PIP := $(PY) -m pip
REQ ?= requirements.txt

# ------------ targets ---------------------------------
.PHONY: env dev lint test dash clean

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

lint:           ## run ruff + black (check only)
	$(PY) -m ruff check .

test:           ## run pytest suite
	$(PY) -m pytest -q

dash:           ## open latest W&B run in browser (Linux)
	@xdg-open "$$( $(PY) common/wandb_utils.py --latest-url )"

clean:          ## remove caches & artefacts
	rm -rf .cache __pycache__ */__pycache__ *.egg-info $(VENV)
