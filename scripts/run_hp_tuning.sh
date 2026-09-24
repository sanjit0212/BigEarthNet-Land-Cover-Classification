#!/usr/bin/env bash
# Week 2 item 10: minimal hyperparameter tuning on S2 VALIDATION only (never
# test). Small documented grid -- a rigor item as much as an accuracy one,
# since neither learner had ever been tuned at all before this.
#   LR:  regParam in {0.001, 0.1}      (0.01 baseline already run: s2_lr_full)
#   RF:  (numTrees=50,maxDepth=15), (numTrees=100,maxDepth=10)  (50/10 baseline: s2_rf_full)
set -euo pipefail

JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features_full"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"

run_lr() {
  local reg="$1" run_id="$2"
  echo "=== [$(date +%H:%M:%S)] LR regParam=$reg run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime s2 --fold "" --learner lr --reg-param "$reg" \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6
  echo "=== [$(date +%H:%M:%S)] DONE run_id=$run_id ==="
}

run_rf() {
  local trees="$1" depth="$2" run_id="$3"
  echo "=== [$(date +%H:%M:%S)] RF numTrees=$trees maxDepth=$depth run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime s2 --fold "" --learner rf --num-trees "$trees" --max-depth "$depth" \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6
  echo "=== [$(date +%H:%M:%S)] DONE run_id=$run_id ==="
}

run_lr "0.001" "hp_tune_lr_reg0.001"
run_lr "0.1"   "hp_tune_lr_reg0.1"
run_rf "50" "15"  "hp_tune_rf_t50d15"
run_rf "100" "10" "hp_tune_rf_t100d10"

echo "=== HYPERPARAMETER TUNING GRID COMPLETE (4/4) ==="
