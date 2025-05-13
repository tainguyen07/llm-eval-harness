.PHONY: install lint format test test-fast cov build clean docs serve help

PYTHON ?= python
SRC := src
TESTS := tests

help:
	@echo "llm-eval-harness — targets:"
	@echo "  install     Install package + dev extras."
	@echo "  lint        Run ruff + mypy."
	@echo "  format      Run black + isort + ruff format."
	@echo "  test        Run pytest with coverage."
	@echo "  test-fast   Run pytest without coverage, parallel."
	@echo "  cov         Open HTML coverage report."
	@echo "  build       Build sdist + wheel."
	@echo "  clean       Remove build, cache, and report artifacts."
	@echo "  docs        Build Sphinx docs (placeholder)."

install:
	$(PYTHON) -m pip install -e ".[dev,all]"

lint:
	ruff check $(SRC) $(TESTS)
	mypy $(SRC)

format:
	black $(SRC) $(TESTS)
	isort $(SRC) $(TESTS)
	ruff check --fix $(SRC) $(TESTS)
	ruff format $(SRC) $(TESTS)

test:
	pytest

test-fast:
	pytest -n auto --no-cov -q

cov:
	@echo "Coverage report: htmlcov/index.html"

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf reports runs .llm-eval

docs:
	@echo "Docs stub — see docs/ARCHITECTURE.md"