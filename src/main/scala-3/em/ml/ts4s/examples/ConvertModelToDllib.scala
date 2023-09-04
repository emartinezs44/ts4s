package em.ml.ts4s.examples

import em.ml.ts4s.dllib.nlp.models.RobertaForSequenceClassification

object ConvertModelToDllib {
  def main(args: Array[String]): Unit = {
    val onnxPath = args(0)
    val outputClasses = args(1)
    
    def robertaModel =
      new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768, nBlock = 1, useLoraInMultiHeadAtt = true)

    /** Include the number of output classes. This process creates a new RobertaForSequenceClassification from a pretrained Roberta Encoder
      */
    robertaModel.convertModelFromOnnx("model_encoder_roberta_v2.onnx", 4, "model_lora.bigdl", "weights_lora.bigdl")
  }
}
