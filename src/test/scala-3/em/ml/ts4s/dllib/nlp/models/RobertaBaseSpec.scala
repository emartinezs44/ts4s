package em.ml.ts4s.dllib.nlp.models

import org.junit.Test

class RobertaBaseSpec {
  @Test
  def createModelFromOnnx(): Unit = {
    def robertaModel = new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768, nBlock = 1)
    robertaModel.convertModelFromOnnx("/tmp/model_encoder_roberta_v2.onnx", 2, "/tmp/model1.bigdl", "/tmp/weights1.bigdl")
  }
}
