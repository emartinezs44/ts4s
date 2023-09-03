package em.ml.ts4s.dllib.nlp.models

import em.ml.ts4s.dllib.util.LoadWeightsAndBiases
import org.junit.Test

class ConvertModelSpec {

  @Test
  def convertModelFromOnnxExpect(): Unit = {
    val jsonStreamEncoderPath = getClass.getClassLoader.getResourceAsStream("models/roberta_base.json")
    val jsonStreamHeadPath = getClass.getClassLoader.getResourceAsStream("models/classification_head.json")
    val tensors = LoadWeightsAndBiases.loadFromOnnx("/tmp/model_encoder_roberta_v2.onnx", jsonStreamEncoderPath, Some(jsonStreamHeadPath))
    tensors
  }
}
