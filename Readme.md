# ts4s

### Introduction 
---
This is a very experimental project intended to train language models, (and maybe in the future vision transformers) using distributed computing infrastructures like Apache Spark.

Due to the explosion of the based in attention deep learning models, this initiative has the objective of exploring the possibilities and deficiencies that exist in the Scala and Apache Spark community comparing to all the incredible advances in other source projects and frameworks. 

This project contains a RoBERTa implementation that is capable or being executed in Spark clusters taking as input a pretrained model from the HuggingFace hub. It uses BigDL dllib as deep learning framework and is based in Spark primitives only. 

It runs only in CPU using MKL using FP32 precision due to the underlying deep learning ramework but changes available in other initiatives like Int4 quantification and Lora adaptation will be included,

### Where to begin
---
To start this journey first you will need to convert the example pre-trained model from https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne to onnx. You can use the script in the python directory, including as parameters the path where you want to store the file:

```
python convert.py roberta_encoder.onnx
```

You must install the dependencies before launching the process. If you use pip you must have the rust compiler installed.

### Obtaining BigDL dllib
---
This project is created using Scala 3, it is necessary to have a Spark distribution for Scala 2.13. You can download from https://spark.apache.org/downloads.html. But for BigDL there is not official Scala 2.13 release, so you need to clone the forked repo https://github.com/emartinezs44/BigDL and build from sources:

```
cd scala
./make-dist.sh -P scala_2.13
```

Note that you will need Maven to build BigDL. The pom.xml is changed to build only the dllib project and other needed dependencies. After the compilation you will have in the dllib/target the **bigdl-dllib-spark_3.2.3-2.3.0-SNAPSHOT-jar-with-dependencies.jar**.

### Creating the ts4s artifacts
---
There is a sbt task to copy the necessary dependenciaes that you will need for submitting the jar to the Spark cluster.

```
sbt package copy
```
The ts4s is cretated in **target/scala-3.3.0/** and the required dependencies in the **dist** floder.

### Submiting to the Spark cluster
First, the model must be transformed from onnx to bigdl format. There is a script to load the onnx file to a bigdl:

````