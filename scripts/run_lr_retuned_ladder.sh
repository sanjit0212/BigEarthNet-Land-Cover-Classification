#!/usr/bin/env bash
# Week 2 follow-up: hyperparameter tuning found LR's default regParam=0.01 is
# suboptimal (regParam=0.001 beats it by 2.93 points on S2 validation). Rerun
# the full LR ladder at the tuned value so every reported LR number reflects
# the tuned config, not an untuned default. S2 already covered by the tuning
# grid's hp_tune_lr_reg0.001 run -- this covers S1, S3 (5 folds), S4 (10 countries).
set -euo pipefail

JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features_full"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"
COUNTRIES=(Finland Portugal Serbia Lithuania Ireland Austria Belgium Switzerland Luxembourg Kosovo)

run_job() {
  local regime="$1" fold="$2" run_id="$3"
  echo "=== [$(date +%H:%M:%S)] regime=$regime fold='$fold' learner=lr regParam=0.001 run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime "$regime" --fold "$fold" --learner lr --reg-param 0.001 \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6
  echo "=== [$(date +%H:%M:%S)] DONE run_id=$run_id ==="
}

run_job "s1" "" "lr_tuned_s1"

for fold in 0 1 2 3 4; do
  run_job "s3" "$fold" "lr_tuned_s3_fold${fold}"
done

for country in "${COUNTRIES[@]}"; do
  run_job "s4" "$country" "lr_tuned_s4_${country}"
done

echo "=== LR RETUNED LADDER COMPLETE (16/16 jobs) ==="
