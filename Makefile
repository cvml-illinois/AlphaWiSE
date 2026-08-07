# Every check this repository makes is defined here, so a human and any automation run
# the same thing.
.PHONY: help smoke lint

PYTHON ?= python

help:
	@echo "smoke       the whole pipeline on synthetic data: the integration test (tests/smoke.sh)"
	@echo "lint        ruff (when installed) plus the tracked-artifact and debug-hook greps"

smoke:
	PYTHON=$(PYTHON) bash tests/smoke.sh

# ruff is optional; the greps below need only git and grep.
lint:
	@if $(PYTHON) -c "import ruff" >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check alphawise scripts; \
	else \
		echo "ruff not installed; style check skipped"; \
	fi
	@! git ls-files | grep -E '__pycache__|\.pyc$$|\.DS_Store$$' \
		|| { echo "build artifacts are tracked"; exit 1; }
	@! grep -rn "import pdb\|breakpoint()" alphawise/ scripts/ \
		|| { echo "debugger hooks are present"; exit 1; }
