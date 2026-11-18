#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$MODE" != "audit" && "$MODE" != "full" ]]; then
  echo "Usage: bash run_publication_study.sh [audit|full]" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import importlib

required = ("dolfin", "numpy", "pandas", "scipy", "matplotlib")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit(
        "The active Python environment is not compatible with this package:\n- "
        + "\n- ".join(missing)
        + "\nCreate and activate environment.yml before running this command."
    )
PY

"$PYTHON_BIN" tools/verify_package.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m unittest discover \
  -s tests -p 'test_*_invariants.py' -v

if [[ "$MODE" == "audit" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/reproduction_output/audit}"
  mkdir -p "$OUTPUT_ROOT/processed" "$OUTPUT_ROOT/figures"
  "$PYTHON_BIN" src/analyze_ensemble.py \
    --input-dir results/raw \
    --output-dir "$OUTPUT_ROOT/processed" \
    --figure-dir "$OUTPUT_ROOT/figures" \
    --mesh-study-dir results/verification \
    --expected-config config/publication_study.json \
    --expected-pool-seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110 4111 4112
  "$PYTHON_BIN" tools/compare_processed_outputs.py \
    results/processed "$OUTPUT_ROOT/processed"
  echo "Packaged results, invariant tests, and regenerated analysis all verified."
  exit 0
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/reproduction_output/full}"
mkdir -p "$OUTPUT_ROOT/raw" "$OUTPUT_ROOT/processed" "$OUTPUT_ROOT/figures"

"$PYTHON_BIN" src/run_ensemble.py \
  --config config/publication_study.json \
  --output-dir "$OUTPUT_ROOT/raw" \
  --pool-seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110 4111 4112 \
  --policies fixed random feedback \
  --jobs 1 \
  --representative-xdmf

"$PYTHON_BIN" src/analyze_ensemble.py \
  --input-dir "$OUTPUT_ROOT/raw" \
  --output-dir "$OUTPUT_ROOT/processed" \
  --figure-dir "$OUTPUT_ROOT/figures" \
  --mesh-study-dir results/verification \
  --expected-config config/publication_study.json \
  --expected-pool-seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110 4111 4112

echo "Full 36-run reproduction completed under: $OUTPUT_ROOT"
