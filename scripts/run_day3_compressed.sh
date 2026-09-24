#!/usr/bin/env bash
# Compressed Day-3 sequence per SPRINT.md Sec 6 failure protocol: S3 (5 tile
# folds) dropped to save ~2h; S1/S2/S4 still make the split-audit argument,
# and S4 (leave-country-out) is the one that matters most.
#   1. RF on S2 (completes Table 1 / E1)
#   2. LR on S1 (random split)
#   3. LR on S4 x 10 countries (leave-country-out)
# S2/LR was already run separately (e1_lr_s2_full).
set -euo pipefail

JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features_full"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"
COUNTRIES=(Finland Portugal Serbia Lithuania Ireland Austria Belgium Switzerland Luxembourg Kosovo)

run_job() {
  local regime="$1" fold="$2" learner="$3" run_id="$4"
  echo "=== [$(date +%H:%M:%S)] regime=$regime fold='$fold' learner=$learner run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime "$regime" --fold "$fold" --learner "$learner" \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6
  echo "=== [$(date +%H:%M:%S)] DONE run_id=$run_id ==="
}

run_job "s2" ""  "rf" "e1_rf_s2_full"
run_job "s1" ""  "lr" "e3_full_lr_s1"

for country in "${COUNTRIES[@]}"; do
  run_job "s4" "$country" "lr" "e3_full_lr_s4_${country}"
done

echo "=== ALL DAY-3 COMPRESSED RUNS COMPLETE ==="
