##Title Ts4s

Subtitle Introduction
This is a very experimental project intended to train language models, (and maybe in the future vision transformers) using distributed computing infrastructures like Apache Spark.

Due to the explosion of the based in attention deep learning models, this initiative has the objective of exploring the possibilities and deficiencies that exist in the Scala and Apache Spark community
following all the incredible advances in other other source projects.

This project contains a RoBERTa implementation that is capable or being executed in Spark Clusters taking as input a pretrained model from the HuggingFace hub. It uses BigDL dllib as deep learning framework and
is based in Spar primitives only.

It runs only in CPU using MKL using FP32 precision, in the future changes available in other initiatives like Int4 quantification and Lora adaptation will be included.

Usage

