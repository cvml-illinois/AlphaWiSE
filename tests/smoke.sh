#!/usr/bin/env bash
# End-to-end smoke test: the whole pipeline on a generated synthetic corpus.
#
# Generates a corpus with the same on-disk shape as AudioSet, trains briefly, fuses,
# evaluates, renders the tables, and checks that every stage produced the files it
# declares. Needs the four assets in assets/ (see assets/README.md), no GPU, no AudioSet,
# and no external checkpoints. This is the check to run first on a new machine.
#
#   bash tests/smoke.sh            # or: make smoke
#   PYTHON=.venv/bin/python bash tests/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

echo "=== import sanity ==="
"$PYTHON" - <<'PY'
import alphawise, alphawise.naming, alphawise.fusion.blend, alphawise.config
print("alphawise imports OK")
PY

echo "=== every artifact's commands parse (no step drift) ==="
"$PYTHON" -m alphawise.reproduce --all --dry-run

echo "=== smoke artifact: generate, train, fuse, evaluate, render ==="
"$PYTHON" -m alphawise.reproduce --artifact smoke --verify

echo "=== smoke OK ==="
