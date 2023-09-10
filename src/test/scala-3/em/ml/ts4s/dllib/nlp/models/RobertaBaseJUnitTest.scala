package em.ml.ts4s.dllib.nlp.models

import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import org.junit.Test
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
        Array(1, inputArray1.length /*RobertaBase.seqLen*/ )
      )

    //val masks: Tensor[Float] = Tensor[Float](1, 1, 1, seqLen).fill(1.0f)
    /*
    val masks: Tensor[Float] = Tensor[Float](
      Array.fill(inputArray0.length)(1.0f) ++ Array.fill(
        RobertaBase.seqLen - inputArray0.length
      )(0f),
      Array(1, 1, 1, inputArray1.length /*RobertaBase.seqLen*/ )
    )*/

    /*
    val masksTwo: Tensor[Float] = Tensor[Float](
      Array.fill(inputArray0.length)(0.0f) ++ Array.fill(
        RobertaBase.seqLen - inputArray0.length
      )(Float.NegativeInfinity),
      Array(1, 1, 1, inputArray1.length /*RobertaBase.seqLen*/ )
    )*/

    val masksTwo: Tensor[Float] = Tensor[Float](
      Array.fill(inputArray0.length)(1.0f) ++ Array.fill(
        seqLen - inputArray0.length
      )(0f),
      Array(1, 1, 1, inputArray1.length /*RobertaBase.seqLen*/ )
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
    //inputIds
  }

  @Test
  def checkRoberta0(): Unit = {
    val seqLen            = 514
    val bertBaseSeqLength = 514
    val hiddenSize        = 768

    /** At the moment NO Seq Len */
    val bertBaseShape = MultiShape(
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
      nBlock = 12,
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

    val model2 = model1//.evaluate()
    model2.build(bertBaseShape)
    val result = model2//.evaluate()


    val t = System.nanoTime
    result.forward(input)
    val duration = (System.nanoTime - t) / 1e9d

    println("D: " + duration)
    val t0 = System.nanoTime
    result.forward(input)
    val duration1 = (System.nanoTime - t0) / 1e9d
    println("D: " + duration1)

    val t1 = System.nanoTime
    result.forward(input)
    val duration2 = (System.nanoTime - t1) / 1e9d
    println("D: " + duration2)

    val t2 = System.nanoTime
    result.forward(input)
    val duration3 = (System.nanoTime - t2) / 1e9d
    println("D: " + duration3)

    val t3 = System.nanoTime
    result.forward(input)
    val duration4 = (System.nanoTime - t3) / 1e9d
    println("D: " + duration4)

    val t4 = System.nanoTime
    result.forward(input)
    val duration5 = (System.nanoTime - t4) / 1e9d
    println("D: " + duration5)

    val t5 = System.nanoTime
    result.forward(input)
    val duration6 = (System.nanoTime - t5) / 1e9d
    println("D: " + duration6)

    val t6 = System.nanoTime
    result.forward(input)
    val duration7 = (System.nanoTime - t6) / 1e9d
    println("D: " + duration7)

    val t8 = System.nanoTime
    result.forward(input)
    val duration9 = (System.nanoTime - t8) / 1e9d
    println("D: " + duration9)

    // val result1 = result.forward(input)

    val result2 = result.forward(input)

    println("backward")
    val t0Back = System.nanoTime
    model1.backward(input, result2)
    val durationt0Back0 = (System.nanoTime - t0Back) / 1e9d
    println("B0: " + durationt0Back0)

    val t1Back = System.nanoTime
    model1.backward(input, result2)
    val durationt0Back01 = (System.nanoTime - t1Back) / 1e9d
    println("B1: " + durationt0Back01)

    val t2Back = System.nanoTime
    model1.backward(input, result2)
    val durationt0Back02 = (System.nanoTime - t2Back) / 1e9d
    println("B2: " + durationt0Back02)

    val t3Back = System.nanoTime
    model1.backward(input, result2)
    val durationt0Back03 = (System.nanoTime - t3Back) / 1e9d
    println("B3: " + durationt0Back03)

    val t4Back = System.nanoTime
    model1.backward(input, result2)
    val durationt0Back04 = (System.nanoTime - t4Back) / 1e9d
    println("B3: " + durationt0Back04)
    /*println("Backward 1")
    model1.backward(input, result2)
    println("Backward 2")
    model1.backward(input, result2)
    println("Backward 3")
    model1.backward(input, result2)
    println("Backward 4")
    model1.backward(input, result2)
    val duration10 = (System.nanoTime - t9) / 1e9d
    println("D: " + duration10 / 5)*/

  }

  @Test
  def softMax(): Unit = {
    val tensor = Tensor[Float](1, 12, 514, 514).rand()

    val softmax = SoftMax[Float]
    val t = System.nanoTime
    val result = softmax.forward(tensor)
    softmax.forward(tensor)
    softmax.forward(tensor)
    softmax.forward(tensor)
    softmax.forward(tensor)
    val d = (System.nanoTime - t) / 1e9d
    println("forward: " + d / 5)

    val t2 = System.nanoTime
    softmax.backward(tensor, result)
    softmax.backward(tensor, result)
    softmax.backward(tensor, result)
    softmax.backward(tensor, result)
    softmax.backward(tensor, result)
    val d2 = (System.nanoTime - t2) / 1e9d
    println("forward: " + d2 / 5)
  }
}
