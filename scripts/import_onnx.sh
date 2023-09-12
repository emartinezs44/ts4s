#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: missing input params!"
    exit 1
fi

INPUT_ONNX_FILE="$1"
OUTPUT_MODEL_PATH="$2"
OUTPUT_WEIGHTS_PATH="$3"
OUTPUT_CLASSES="$4"

JARS_PATH=dist/
JAR_FOLDER=target/scala-3.3.0
MAIN_CLASS=em.ml.ts4s.examples.ConvertModelToDllib

JARS_LIST=$(echo ${JARS_PATH}*.jar | tr ' ' ',')
echo $JARS_LIST

$SPARK_HOME/bin/spark-submit --class ${MAIN_CLASS} \
  --master local \
  --driver-memory 2G \
  --jars $JARS_LIST \
  ${JAR_FOLDER}/ts4s_3-0.0.1.jar \
  --onnxFilePath $INPUT_ONNX_FILE \
  --outputClasses $OUTPUT_CLASSES \
  --outputModelPath $OUTPUT_MODEL_PATH \
  --outputWeightsPath $OUTPUT_WEIGHTS_PATH