#!/usr/bin/env bash
# SPRINT.md Sec 3.2 -- E3 split audit. Runs BinaryRelevanceRunner (learner=lr,
# the workshorse per Sec 2.3) across all four split regimes:
#   S1 random (1 run), S2 official (1 run), S3 leave-tile-out (5 folds),
#   S4 leave-country-out (10 folds) = 17 spark-submit jobs.
#
# Requires the Docker cluster to be up (docker compose up -d) and features +
# splits already in HDFS (see docs/day2_pipeline.md). Run from the repo root:
#   bash scripts/run_e3_audit.sh [full|subset]
set -euo pipefail

TAG_SUFFIX="${1:-full}"   # "full" (default, full 480k) or "subset" (10% dev run)
LEARNER="lr"
JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"

COUNTRIES=(Finland Portugal Serbia Lithuania Ireland Austria Belgium Switzerland Luxembourg Kosovo)

run_job() {
  local regime="$1" fold="$2" run_id="$3"
  echo "=== [$(date +%H:%M:%S)] regime=$regime fold='$fold' run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime "$regime" --fold "$fold" --learner "$LEARNER" \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6 \
    2>&1 | grep -E "^\{|\[done\]|ERROR|Exception|Joined features|Train rows" || true
}

run_job "s1" ""  "e3_${TAG_SUFFIX}_${LEARNER}_s1"
run_job "s2" ""  "e3_${TAG_SUFFIX}_${LEARNER}_s2"

for fold in 0 1 2 3 4; do
  run_job "s3" "$fold" "e3_${TAG_SUFFIX}_${LEARNER}_s3_fold${fold}"
done

for country in "${COUNTRIES[@]}"; do
  run_job "s4" "$country" "e3_${TAG_SUFFIX}_${LEARNER}_s4_${country}"
done

echo "=== E3 audit complete: 17 runs. Compute stats under ${OUTPUT_PATH}/compute/, predictions under ${OUTPUT_PATH}/predictions/ ==="
