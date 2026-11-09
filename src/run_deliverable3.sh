#!/usr/bin/env bash
# Run the Deliverable 3 analysis steps that come after model training (08), in order.
# Stops at the first failure so a broken step cannot silently poison the ones after it.
# Usage: bash src/run_deliverable3.sh
set -euo pipefail
cd "$(dirname "$0")"
for s in 09_evaluate 10_generalization 11_adversarial 12_responsible_ai 13_monitoring 14_write_summary_d3; do
  echo "=============== $s ==============="
  python3 -u "$s.py"
done
echo "=============== build report ==============="
python3 -u ../reports/build_deliverable3.py
echo "ALL_DONE"
