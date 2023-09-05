package em.ml.ts4s.dllib.conf

import com.typesafe.config.Config
import scopt.OptionParser

case class InputParameters(
  inputDatasetPath: String = "",
  inputModelPath: String = "",
  inputWeightsPath: String = "",
  outputModelPath: String = "",
  ouputWeightsPath: String = ""
)

object InputParserInstances {

  val servicerParserInstance = new OptionParser[InputParameters]("App") {
    opt[String]("inputDatasetPath")
      .required()
      .action((value, config) => config.copy(inputDatasetPath = value))
      .text("Missing dataset path")
    opt[String]("inputModelPath")
      .required()
      .action((value, config) => config.copy(inputModelPath = value))
      .text("Missing input model path")
    opt[String]("inputWeightsPath")
      .required()
      .action((value, config) => config.copy(inputWeightsPath = value))
      .text("Missing input weights path")
    opt[String]("outputModelPath")
      .required()
      .action((value, config) => config.copy(inputWeightsPath = value))
      .text("Missing output model path")
    opt[String]("outputWeightsPath")
      .required()
      .action((value, config) => config.copy(inputWeightsPath = value))
      .text("Missing output weights path")
  }
}
