package em.ml.ts4s.dllib.nlp.models

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T

object ExampleData {
  def getSampleTensor1(seqLen: Int) = {

    val inputArray: Array[Float] = Array[Float](7, 20, 39, 27, 10, 39, 30, 21, 17, 15, 7, 20, 39, 27, 10, 39, 30, 21, 17, 15)

    val inputArray1: Array[Float] =
      inputArray ++ Array.fill(seqLen - inputArray.length)(1f)

    val inputArrayTotal: Array[Float] = inputArray1

    val inputIds: Tensor[Float] =
      Tensor[Float](
        inputArrayTotal,
        Array(1, inputArray1.length /*RobertaBase.seqLen*/ )
      )

    val masksTwo: Tensor[Float] = Tensor[Float](
      Array.fill(inputArray.length)(1.0f) ++ Array.fill(
        seqLen - inputArray.length
      )(0f),
      Array(1, 1, 1, inputArray1.length )
    )

    val positionIds: Tensor[Float] =
      Tensor[Float](
        inputArray.indices
          .map(_.toFloat)
          .map(_ + 2)
          .toArray ++ Array.fill(seqLen - inputArray.length)(0f),
        Array(1, seqLen)
      )

    Tensor(Array(inputArray.length).map(_.toFloat), Array(1, 1))

    T(inputIds, positionIds, masksTwo)
  }
}
