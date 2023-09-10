package em.ml.ts4s.examples

import em.ml.ts4s.dllib.conf.{ConverterParameters, InputParameters, InputParserInstances}
import em.ml.ts4s.dllib.nlp.models.RobertaForSequenceClassification

object ConvertModelToDllib {
  def main(args: Array[String]): Unit = {
    InputParserInstances.convetParserInstance.parse(args, ConverterParameters()) match {
      case Some(conf) =>
        def robertaModel =
          new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768)

        /** Include the number of output classes. This process creates a new RobertaForSequenceClassification from a pretrained Roberta Encoder */
        robertaModel.convertModelFromOnnx(conf.onnxFilePath, conf.outputClasses, conf.outputModelPath, conf.outputWeightPath)
      case _ =>
        throw Exception("Error in input data")
    }
  }
}
