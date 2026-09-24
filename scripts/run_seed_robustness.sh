#!/usr/bin/env bash
# Week 2 item 9: seed robustness, minimal per the plan ("report SD once").
#   - S1 split seed: 2 more independent 50/25/25 shuffles (LR, deterministic,
#     cheap) -- pre-empts "your optimistic S1 baseline was a lucky shuffle".
#   - RF training seed: 2 more seeds on S2 (the representative regime) --
#     pre-empts "the RF gap is just training noise".
set -euo pipefail

JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features_full"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"

run_job() {
  local regime="$1" fold="$2" learner="$3" seed="$4" run_id="$5"
  echo "=== [$(date +%H:%M:%S)] regime=$regime fold='$fold' learner=$learner seed=$seed run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime "$regime" --fold "$fold" --learner "$learner" --seed "$seed" \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6
  echo "=== [$(date +%H:%M:%S)] DONE run_id=$run_id ==="
}

run_job "s1" "seed123" "lr" "42" "seed_robust_lr_s1_seed123"
run_job "s1" "seed7"   "lr" "42" "seed_robust_lr_s1_seed7"
run_job "s2" ""        "rf" "123" "seed_robust_rf_s2_seed123"
run_job "s2" ""        "rf" "7"   "seed_robust_rf_s2_seed7"

echo "=== SEED ROBUSTNESS RUNS COMPLETE (4/4) ==="
