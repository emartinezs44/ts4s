package em.ml.ts4s.examples

import em.ml.ts4s.dllib.nlp.models.RobertaForSequenceClassification

object ConvertModelToDllib {
  def main(args: Array[String]): Unit = {
    val onnxPath          = args(0)
    val outputClasses     = args(1)
    val outputPathModel   = args(3)
    val outputPathWeights = args(4)

    def robertaModel =
      new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768, nBlock = 1, useLoraInMultiHeadAtt = true)

    /** Include the number of output classes. This process creates a new RobertaForSequenceClassification from a pretrained Roberta Encoder */
    robertaModel.convertModelFromOnnx(onnxPath, outputClasses.toInt, outputPathModel, outputPathWeights)
  }
}
