#!/usr/bin/env bash
# Reproduce every ScanShield output from the raw data. Usage: bash run_all.sh
#
#   01-06  Deliverable 2: audit, cleaning, EDA, feature sets and splits, baselines, summary
#   07-13  Deliverable 3: feature engineering, model tuning, evaluation and error analysis,
#          generalization, adversarial evasion and defense, responsible AI, monitoring
#   14     writes RESULTS_DELIVERABLE3.md from the saved outputs
#
# Steps 10 and 13 also read the other CICIDS2017 day files from ~/Desktop/CICIDS_dataset
# (see data/README.md). They skip themselves with a message if those files are absent.
set -euo pipefail
cd "$(dirname "$0")/src"
start=$(date +%s)

D2="01_load_and_audit 02_clean 03_eda 04_features_and_split 05_baselines"
D3="07_feature_engineering 08_train_models 09_evaluate 10_generalization 11_adversarial 12_responsible_ai 13_monitoring"

for s in $D2; do
  echo "=== $s ==="
  python "$s.py"
done
end=$(date +%s)
echo "{\"deliverable2_seconds\": $((end-start))}" > ../outputs/metrics/runtime.json
echo "=== 06_write_summary ==="
python 06_write_summary.py

d3start=$(date +%s)
for s in $D3; do
  echo "=== $s ==="
  python "$s.py"
done
d3end=$(date +%s)
echo "{\"deliverable2_seconds\": $((end-start)), \"deliverable3_seconds\": $((d3end-d3start)), \"total_seconds\": $((d3end-start))}" \
  > ../outputs/metrics/runtime.json
echo "=== 14_write_summary_d3 ==="
python 14_write_summary_d3.py

echo "Done in $((d3end-start)) seconds."
