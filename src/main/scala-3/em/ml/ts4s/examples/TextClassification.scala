package em.ml.ts4s.examples

import em.ml.ts4s.dllib.nlp.models.{RobertaBase, RobertaForSequenceClassification}
import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Engine
import em.ml.ts4s.dllib.conf.{InputParameters, InputParserInstances}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.FloatType

import util.chaining.scalaUtilChainingOps
import scala.collection.mutable
import scala.language.postfixOps

def createLabel(data: DataFrame)(labelInputCol: String, labelOutputCol: String) = {
  import org.apache.spark.sql.functions.col
  val st = new StringIndexer()
    .setStringOrderType("alphabetAsc")
    .setInputCol(labelInputCol)
    .setOutputCol(labelOutputCol)

  val stModel = st.fit(data)
  stModel
    .transform(data)
    .withColumn(labelOutputCol, col(labelOutputCol).cast(FloatType))
    .withColumn(labelOutputCol, col(labelOutputCol) + 1.0f)
}

def prepareSamples(dataset: DataFrame, category: String, tokens: String, seqLen: Int): RDD[Sample[Float]] = {
  val trainSetRDD = dataset.select(category, tokens)
  trainSetRDD.rdd
    .map(row => {
      val category: (Option[Float], Option[Array[Float]]) = (Some(row.getAs[Float](0)), None)
      val textInt = {
        val result = row.getAs[mutable.WrappedArray[Int]](1)
        if (result.length > (seqLen - 2))
          result.slice(0, seqLen - 2)
        else result
      }
      val inputSeqLen = textInt.length + 2
      val inputArray0 = Array(0f) ++ textInt
        .map(_.toFloat)
        .toArray[Float] ++ Array(2.0f)
      val featuresArray =
        inputArray0 ++ Array.fill(seqLen - inputSeqLen)(1f)
      // We try to fine tuning starting from 1
      val positionsArray = (2 to seqLen - 1).toArray
      val positionIds: Tensor[Float] =
        Tensor[Float](
          positionsArray.map(_.toFloat) ++ Array(
            positionsArray.last.toFloat,
            positionsArray.last.toFloat
          ),
          Array(seqLen)
        )
      val masks: Tensor[Float] = Tensor[Float](
        Array.fill(inputArray0.length)(1.0f) ++ Array
          .fill(seqLen - inputArray0.length)(0f),
        Array(1, 1, seqLen)
      )
      val featuresTensor = Tensor[Float](featuresArray, Array(seqLen))
      val labelTensor    = Tensor[Float](Tensor(Array(category._1.get), Array(1)))
      Sample(Array(featuresTensor, positionIds, masks), labelTensor)
    })
}

object StartTrainingProcess {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.functions.{split, col}

    val conf = Engine
      .createSparkConf()

    given spark: SparkSession =
      SparkSession
        .builder()
        .master("local[1]")
        .config(conf)
        .getOrCreate()

    // Pass input text
    InputParserInstances.servicerParserInstance.parse(args, InputParameters()) match {
      case Some(inputArgs) =>
        val dfText       = spark.read.text(inputArgs.inputDatasetPath)
        val inputModel   = inputArgs.inputModelPath
        val inputWeights = inputArgs.inputWeightsPath

        val Array(train, validation) = dfText
          .withColumn("array_column", split(col("value"), "\t"))
          .withColumn("cat", col("array_column")(0))
          .withColumn("text", col("array_column")(1))
          .select("cat", "text")
          .randomSplit(Array(80, 20))

        val trainTokenized =
          (((train pipe RobertaForSequenceClassification.tokenizeDataframeColumn)("text", "tokens"))
            pipe createLabel)("cat", "float_cat")

        val validationTokenized =
          (((validation pipe RobertaForSequenceClassification.tokenizeDataframeColumn)("text", "tokens"))
            pipe createLabel)("cat", "float_cat")

        val ro           = new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768, useLoraInMultiHeadAtt = true)
        val trainDF      = prepareSamples(trainTokenized, "float_cat", "tokens", 514)
        val validationDF = prepareSamples(validationTokenized, "float_cat", "tokens", 514)

        Engine.init

        ro.fit(
          checkpointPath = "",
          trainingDataset = trainDF,
          validationDataset = validationDF,
          batchSize = 1,
          epochs = 1,
          outputModelPath = inputArgs.outputModelPath,
          outputWeightsPath = inputArgs.outputWeightsPath,
          categoryColumn = "cat",
          "tokens",
          modelPath = inputModel,
          weightsPath = inputWeights,
          isLora = true
        )

        spark.stop()

      case _ => throw Exception("Error in input data")
    }
  }
}
