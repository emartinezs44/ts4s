package em.ml.ts4s.dllib.nlp.models

import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import org.junit.{Assert, Test}
import org.junit.Assert.*
import com.intel.analytics.bigdl.dllib.nn.{LookupTable, SoftMax}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{MultiShape, Shape, T}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class IntJunitTests {

  def getSampleTensor1(seqLen: Int) = {
    val inputArray0: Array[Float] = Array[Float](7, 20, 39, 27, 10, 39, 30, 21, 17, 15, 7, 20, 39, 27, 10, 39, 30, 21, 17, 15)
    val seqLen = 514
    val inputArray1: Array[Float] =
      inputArray0 ++ Array.fill(seqLen - inputArray0.length)(1f)

    val inputArrayTotal: Array[Float] = inputArray1
    val inputIds: Tensor[Float] =
      Tensor[Float](
        inputArrayTotal,
        Array(1, inputArray1.length /*RobertaBase.seqLen*/)
      )

    val masksTwo: Tensor[Float] = Tensor[Float](
      Array.fill(inputArray0.length)(1.0f) ++ Array.fill(
        seqLen - inputArray0.length
      )(0f),
      Array(1, 1, 1, inputArray1.length /*RobertaBase.seqLen*/)
    )

    val positionIds: Tensor[Float] =
      Tensor[Float](
        inputArray0.indices
          .map(_.toFloat)
          .map(_ + 2)
          .toArray ++ Array.fill(seqLen - inputArray0.length)(0f),
        Array(1, seqLen)
      )

    Tensor(Array(inputArray0.length).map(_.toFloat), Array(1, 1))
    T(inputIds, positionIds, masksTwo)
  }

  @Test
  def checkRoberta0(): Unit = {
    val seqLen = 514
    val bertBaseSeqLength = 514
    val hiddenSize = 768

    /** At the moment NO Seq Len */
    val robertaBaseShape = MultiShape(
      List(
        Shape(1, bertBaseSeqLength),
        Shape(1, bertBaseSeqLength),
        Shape(1, 1, 1, bertBaseSeqLength)
      )
    )

    val input = getSampleTensor1(514)

    val model1 = new RobertaBase[Float](
      vocab = 50262,
      hiddenSize = hiddenSize,
      nBlock = 1,
      nHead = 12,
      intermediateSize = 3072,
      hiddenPDrop = 0.1,
      attnPDrop = 0.1,
      maxPositionLen = bertBaseSeqLength,
      outputAllBlock = false,
      inputSeqLen = bertBaseSeqLength,
      headLayer = None,
      useLoraInMultiHeadAttention = false
    )

    model1.build(robertaBaseShape)
    val result = model1.forward(input)
    result.toTable.get[Tensor[Float]](1) match {
      case Some(output) =>
        Assert.assertArrayEquals(output.size(), Array(1, seqLen, hiddenSize))
    }
  }
}
  
