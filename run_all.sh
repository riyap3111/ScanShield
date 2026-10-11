#!/usr/bin/env bash
# Reproduce every Deliverable 2 output from the raw data. Usage: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/src"
start=$(date +%s)
for s in 01_load_and_audit 02_clean 03_eda 04_features_and_split 05_baselines; do
  echo "=== $s ==="
  python "$s.py"
done
end=$(date +%s)
echo "{\"total\": \"$((end-start)) seconds\"}" > ../outputs/metrics/runtime.json
echo "=== 06_write_summary ==="
python 06_write_summary.py
echo "Done in $((end-start)) seconds."
