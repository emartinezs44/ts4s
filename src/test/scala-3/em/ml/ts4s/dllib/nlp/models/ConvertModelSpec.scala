package em.ml.ts4s.dllib.nlp.models

import em.ml.ts4s.dllib.util.LoadWeightsAndBiases
import org.junit.Test
import org.junit.Assert.*
import org.junit.{Assert, Test}

class ConvertModelSpec {

  @Test
  def convertModelFromOnnxExpect(): Unit = {
    val onnxPath = ""
    if (!onnxPath.isEmpty)
       val jsonStreamEncoderPath = getClass.getClassLoader.getResourceAsStream("models/roberta_base.json")
       val jsonStreamHeadPath    = getClass.getClassLoader.getResourceAsStream("models/classification_head.json")
       val tensors               = LoadWeightsAndBiases.loadFromOnnx(onnxPath, jsonStreamEncoderPath, Some(jsonStreamHeadPath))
       Assert.assertTrue(tensors.length == 199)
  }
}
