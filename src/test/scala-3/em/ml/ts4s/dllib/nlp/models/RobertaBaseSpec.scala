package em.ml.ts4s.dllib.nlp.models

import org.junit.{Assert, Test}
import com.intel.analytics.bigdl.dllib.utils.SingleShape
class RobertaBaseSpec {
  @Test
  def createModelFromOnnx(): Unit = {
    val onnxPath = ""

    def robertaModel = new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768, nBlock = 1)

    if (!onnxPath.isEmpty)
       val module = robertaModel.convertModelFromOnnx(onnxPath, 2, "/tmp/model2.bigdl", "/tmp/weights2.bigdl")
       Assert.assertEquals(module.parameters()._1.length, 200)
  }
}
