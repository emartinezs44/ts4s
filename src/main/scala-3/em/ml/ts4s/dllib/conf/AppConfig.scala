package em.ml.ts4s.dllib.conf

import com.typesafe.config.Config
import scopt.OptionParser

case class ConverterParameters(
  onnxFilePath: String = "",
  outputClasses: Int = 0,
  outputModelPath: String = "",
  outputWeightPath: String = ""
)

case class InputParameters(
  inputDatasetPath: String = "",
  inputModelPath: String = "",
  inputWeightsPath: String = "",
  outputModelPath: String = "",
  outputWeightsPath: String = ""
)

object InputParserInstances {

  val convetParserInstance = new OptionParser[ConverterParameters]("Converter") {
    opt[String]("onnxFilePath")
      .required()
      .action((value, config) => config.copy(onnxFilePath = value))
      .text("Missing onnx path")

    opt[Int]("outputClasses")
      .required()
      .action((value, config) => config.copy(outputClasses = value))
      .text("Missing output class number")
    opt[String]("outputModelPath")
      .required()
      .action((value, config) => config.copy(outputModelPath = value))
      .text("Missing output model path")
    opt[String]("outputWeightsPath")
      .required()
      .action((value, config) => config.copy(outputWeightPath = value))
      .text("Missing output weights path")
  }

  val servicerParserInstance = new OptionParser[InputParameters]("TextClassification") {
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
      .action((value, config) => config.copy(outputModelPath = value))
      .text("Missing output model path")
    opt[String]("outputWeightsPath")
      .required()
      .action((value, config) => config.copy(outputWeightsPath = value))
      .text("Missing output weights path")
  }
}
