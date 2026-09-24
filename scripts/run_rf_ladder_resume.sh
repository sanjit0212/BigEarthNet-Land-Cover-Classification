#!/usr/bin/env bash
# Resume of run_rf_ladder.sh -- S1, S3 (all 5 folds), and S4 for
# Finland/Portugal/Serbia/Lithuania/Ireland already completed (verified 19/19
# labels each in HDFS). Austria was interrupted mid-run (only 6/19 labels
# written) so it's rerun cleanly (.mode("overwrite") handles the partial
# leftovers); Belgium/Switzerland/Luxembourg/Kosovo were never started.
set -euo pipefail

JAR="/workspace/target/scala-2.12/bigearthnet-classifier-assembly-0.1.jar"
FEATURES_PATH="hdfs://namenode:9000/bigearthnet/features_full"
SPLITS_PATH="hdfs://namenode:9000/bigearthnet/splits/splits.parquet"
OUTPUT_PATH="hdfs://namenode:9000/bigearthnet/results"
COUNTRIES=(Austria Belgium Switzerland Luxembourg Kosovo)

run_job() {
  local regime="$1" fold="$2" run_id="$3"
  echo "=== [$(date +%H:%M:%S)] regime=$regime fold='$fold' learner=rf run_id=$run_id ==="
  MSYS_NO_PATHCONV=1 docker exec spark-master /spark/bin/spark-submit \
    --class multilabel.BinaryRelevanceRunner \
    --master spark://spark-master:7077 \
    --executor-memory 5g --executor-cores 3 --total-executor-cores 6 --driver-memory 2g \
    "$JAR" \
    --regime "$regime" --fold "$fold" --learner rf \
    --features-path "$FEATURES_PATH" --splits-path "$SPLITS_PATH" \
    --output-path "$OUTPUT_PATH" --run-id "$run_id" --thread-pool-size 6
  echo "=== [$(date +%H:%M:%S)] DONE run_id=$run_id ==="
}

for country in "${COUNTRIES[@]}"; do
  run_job "s4" "$country" "e3_full_rf_s4_${country}"
done

echo "=== RF LADDER RESUME COMPLETE (5/5 remaining jobs) ==="
