#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: missing input params!"
    exit 1
fi

INPUT_DATASET="$1"
INPUT_MODEL_PATH="$2"
INPUT_WEIGHTS_PATH="$3"
OUTPUT_MODEL_PATH="$4"
OUTPUT_WEIGHTS_PATH="$5"
BATCH_SIZE="$6"

DRIVER_MEMORY=14G
EXECUTOR_MEMORY=1G
JARS_PATH=dist/
JAR_FOLDER=target/scala-3.3.0
MAIN_CLASS=em.ml.ts4s.examples.StartTrainingProcess
MAX_CORES=1
EXECUTOR_CORES=1

JARS_LIST=$(echo ${JARS_PATH}*.jar | tr ' ' ',')

$SPARK_HOME/bin/spark-submit --class ${MAIN_CLASS} \
  --master local[1]  --conf spark.driver.extraJavaOptions="-Xss512m -XX:+UseG1GC"  \
  --conf spark.cores.max=${MAX_CORES} \
  --conf spark.task.maxFailures=1 \
  --conf spark.executor.memory=${EXECUTOR_MEMORY} \
  --driver-memory ${DRIVER_MEMORY} \
  --executor-cores ${EXECUTOR_CORES} \
  --executor-memory ${EXECUTOR_MEMORY} \
  --jars  $JARS_LIST  \
  ${JAR_FOLDER}/ts4s_3-0.0.1.jar \
  --inputDatasetPath ${INPUT_DATASET} \
  --inputModelPath ${INPUT_MODEL_PATH} \
  --inputWeightsPath ${INPUT_WEIGHTS_PATH} \
  --outputModelPath ${OUTPUT_MODEL_PATH} \
  --outputWeightsPath ${OUTPUT_WEIGHTS_PATH} \
  --batchSize ${BATCH_SIZE}
