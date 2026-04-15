.PHONY: help init test baseline-save baseline-diff baseline-clear archive archive-dry clean

PYTHON ?= python3
VENV   := .venv

help:
	@echo "Targets:"
	@echo "  init            - create .venv + install project with [dev] extras"
	@echo "  test            - run the full pytest suite"
	@echo "  baseline-save   - snapshot current pytest failures as baseline"
	@echo "  baseline-diff   - run pytest, print only new red / newly green vs baseline"
	@echo "  baseline-clear  - remove the stored baseline"
	@echo "  archive         - archive workspaces older than 30 days"
	@echo "  archive-dry     - preview what archive would do without touching anything"
	@echo "  clean           - remove venv, caches, baselines"

init:
	PYTHON=$(PYTHON) scripts/setup.sh

test:
	$(VENV)/bin/python -m pytest tests/

baseline-save:
	scripts/pytest_baseline.sh save

baseline-diff:
	scripts/pytest_baseline.sh diff

baseline-clear:
	scripts/pytest_baseline.sh clear

archive:
	scripts/archive_workspaces.sh

archive-dry:
	scripts/archive_workspaces.sh --dry-run

clean:
	rm -rf $(VENV) .pytest_cache .pytest_baseline
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
