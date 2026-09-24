#!/usr/bin/env bash
# Week 1 item 5 of the strengthening plan: run S3 leave-tile-out (5 folds, LR,
# full 480,038-patch data) -- built in build_splits.py but never executed.
set -euo pipefail

JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features_full"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"

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

for fold in 0 1 2 3 4; do
  run_job "s3" "$fold" "lr" "e3_full_lr_s3_fold${fold}"
done

echo "=== S3 AUDIT COMPLETE (5/5 folds) ==="
