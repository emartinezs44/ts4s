# ts4s

### Introduction 
---
This is a very experimental project intended to train language models, (and maybe in the future vision transformers) using distributed computing infrastructures like Apache Spark.

Due to the explosion of the based in attention deep learning models, this initiative has the objective of exploring the possibilities and deficiencies that exist in the Scala and Apache Spark enviroments comparing to all the incredible advances in other source projects and frameworks.

This project contains a RoBERTa implementation that is capable or being executed in Spark clusters taking as input a pretrained model from the HuggingFace hub. It uses BigDL dllib as deep learning framework and is based in Spark primitives only.

It runs only in CPU using MKL using FP32 precision due to the underlying deep learning ramework but changes available in other projects like Int4 quantification and Lora adaptation could be added in the future.

### Where to begin
---
To start this journey first you will need to convert the example pre-trained model from https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne to onnx. You can use the script in the python directory, including as parameters the path where you want to store the onnx file:

```
python convert.py roberta_encoder.onnx
```

You must install the dependencies before launching the process. If you use pip you must have the rust compiler installed.

### Obtaining BigDL dllib
---
This project is created using Scala 3.3 therefore it is necessary to have a Spark distribution for Scala 2.13. You can download from https://spark.apache.org/downloads.html.

 There is not official BidDL Dllib for Scala 2.13 release, so you need to clone the forked repo https://github.com/emartinezs44/BigDL and build from sources:

```
cd scala
./make-dist.sh -P scala_2.13
```

Note that you will need **Maven** to build BigDL. The pom.xml is changed to build only the dllib project and other needed dependencies. After the compilation you will have in the dllib/target the **bigdl-dllib-spark_3.2.3-2.3.0-SNAPSHOT-jar-with-dependencies.jar**.

### Creating the ts4s artifacts
---
There is a sbt task to copy the necessary dependencies that you will need for submitting the jar to the Spark cluster.

```
sbt package copy
```
The ts4s is cretated in **target/scala-3.3.0/** and the required dependencies in the **dist** folder.

### Get the dataset
You can follow the instructions to download the dataset in https://huggingface.co/datasets/PlanTL-GOB-ES/MLDoc. Here you will find how to create the dataset for testing purposes.

Or you can use another spanish based dataset. The example included in the submit.sh script reads a file with the following format:

[Label][tab][Plain Text...]

Example:
MCAT	 MADRID, 29 may (Reuter) - .......

Follow the code in the **examples/TextClassification.scala** to see how the Spark dataframe is created and apply the changes for your own datasets.

### Submiting to the Spark cluster

First, the model must be transformed from onnx to bigdl format. There is a script to load the onnx file and generate the classificaton model in bidl format.  Set the SPARK_HOME environment variable first and excute the script including as parameters:
 - Location of the onnx file.
 - Location of the biddl model file.
 - Location of the bigdl weights file.
 - Number of output classes for you classification case
```
./scripts/import_onnx.sh onnx_roberta_model.onnx model_test.bigdl weights_test.bigdl 4
```

After that, you can throw your model to your Spark cluster passing as parameters:
- Input dataset(Note that with the format explained avobe)
- Bigdl model file.
- Bigdl weights file.
- Ouput model path.
- Weights output path.
- Batch size. The buth size must be a number divisible by the number of max number of cores.

Update the submit.sh script in order to adapt the paths of the neccessary paths, memory of the driver and executors, the number of cores and cores per executor.

Once the changes are applied, run the submit script:

```
scripts/submit.sh spanish.train.str.1K.txt output_model_test.bigdl output_weights_path.bigdl output_model_test2.bigdl output_weights_path.bigdl 8
```

### NOTES:

This project at the training phase consumes a lot of heap, so you must tune your executors memory in order to increase the batch size. If you try to do in your laptop, consider that this framework is a normal Spark application that start all the Spark environment, cache the model and use the block manager to reduce the weights every iteration and it is very slow comparing to other approachs running locally.