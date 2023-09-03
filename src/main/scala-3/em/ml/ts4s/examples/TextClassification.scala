package em.ml.ts4s.examples

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import em.ml.ts4s.dllib.nlp.models.{RobertaBase, RobertaForSequenceClassification}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable

def prepareSamples(dataset: DataFrame, category: String, tokens: String, seqLen: Int): RDD[Sample[Float]] = {
  /** These are the two dataframe fields to create the input tensors * */
  val trainSetRDD = dataset.select(category, tokens)
  trainSetRDD.rdd
    .map(row => {
      val category: (Option[Float], Option[Array[Float]]) =
        (Some(row.getAs[Float](0)), None)

      val textInt = {
        val result = row.getAs[mutable.WrappedArray[Int]](1)
        if (result.length > (seqLen - 2))
          result.slice(0, seqLen - 2)
        else result
      }

      val inputSeqLen = (textInt.length + 2)
      val inputArray0 = Array(0f) ++ textInt
        .map(_.toFloat)
        .toArray[Float] ++ Array(2.0f)
      val featuresArray =
        inputArray0 ++ Array.fill(seqLen - inputSeqLen)(1f)

      // We try to fine tuning starting from 1
      val positionsArray =
        (2 to seqLen - 1).toArray

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

      val featuresTensor =
        Tensor[Float](featuresArray, Array(seqLen))

      val labelTensor =
        Tensor[Float](Tensor(Array(category._1.get), Array(1)))

      Sample(Array(featuresTensor, positionIds, masks), labelTensor)
    })
}

object ConvertFromOnnx {
  @main def run = {
    val ro = new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768)
    ro.convertModelFromOnnx(onnxModelPath = "/tmp/model_encoder_roberta_v2.onnx", outputClasses = 4, modelStorePath = "/tmp/model.bigdl", weightsStorePath = "/tmp/weights.bigdl")
  }
}

object  StartTrainingProcess {
  @main def start = {
    given spark : SparkSession = SparkSession.builder().master("local[*]").getOrCreate()

    val dfText = spark.read.text(" /Users/e049627/Development/turing/MLDoc/spanish.train.str.1K.txt")
    dfText.rdd.map(row => row.getAs[String](0).split("\t")).take(5).foreach(el => println(el.toList))

    //val ro = new RobertaForSequenceClassification(seqLen = 514, hiddenSize = 768)

  }
}


