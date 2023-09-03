package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.common.TensorOperation
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class LayerNormUpdated[T: ClassTag](
  val nOutput: Int = 768,
  val eps: Double = 1e-5
)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  //val weight = Tensor.ones[T](nOutput).view(1, nOutput)
  val weight = Tensor[T](nOutput).view(1, nOutput).rand()
  val bias   = Tensor[T](nOutput).view(1, nOutput).rand()

  var gradWeight: Tensor[T] = Tensor[T]()
  var gradBias: Tensor[T]   = Tensor[T]()

  var y: Tensor[T]         = null
  var divInput1: Tensor[T] = null
  var divInput2: Tensor[T] = null
  var sqrtInput: Tensor[T] = null

  private def optimzedOperation(
    input1: Tensor[T],
    input2: Tensor[T],
    operation: String
  ) = {
    val dim = input1.dim()
    val kk  = Array.fill[Int](dim - 1)(1)
    var m   = 0
    var cnt = 0

    while (kk(0) < input1.size(1) + 1) {
      cnt += 1
      if (cnt < input1.dim() - 1) {
        m = 1
        while (m < kk.size) {
          kk(m) = 1
          m += 1
        }
        while (kk(1) < input1.size(2) + 1) {
          cnt += 1
          if (cnt < input1.dim() - 1) {
            m = 2
            while (m < kk.size) {
              kk(m) = 1
              m += 1
            }
            while (kk(2) < input1.size(3) + 1) {
              cnt += 1
              if (cnt < input1.dim() - 1) {} else {
                if (operation == "-") {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .sub(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else if (operation == "/") {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .div(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else if (operation == "*") {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .mul(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .add(input2.valueAt(kk(0), kk(1), kk(2), 1))
                }
              }
              kk(2) += 1
              cnt = 2
            }
          } else {
            if (operation == "-") {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .sub(input2.valueAt(kk(0), kk(1), 1))
            } else if (operation == "/") {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .div(input2.valueAt(kk(0), kk(1), 1))
            } else if (operation == "*") {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .mul(input2.valueAt(kk(0), kk(1), 1))
            } else {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .add(input2.valueAt(kk(0), kk(1), 1))
            }
          }
          kk(1) += 1
          cnt = 1
        }
      } else {
        if (operation == "-") {
          input1.narrow(1, kk(0), 1).sub(input2.valueAt(kk(0), 1))
        } else if (operation == "/") {
          input1.narrow(1, kk(0), 1).div(input2.valueAt(kk(0), 1))
        } else if (operation == "*") {
          input1.narrow(1, kk(0), 1).mul(input2.valueAt(kk(0), 1))
        } else {
          input1.narrow(1, kk(0), 1).add(input2.valueAt(kk(0), 1))
        }
      }
      kk(0) += 1
      cnt = 0
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val u   = input.sum(dim).div(ev.fromType(input.size(dim)))
    divInput1 = TensorOperation.subTensor(input.clone(), u)
    val square = divInput1.clone().square()
    val s      = square.sum(square.dim()).div(ev.fromType(square.size(square.dim())))
    sqrtInput = s.add(ev.fromType(eps))
    divInput2 = sqrtInput.clone().sqrt()
    y = TensorOperation.divTensor(divInput1.clone(), divInput2)
    output = y.clone().cmul(weight).add(bias)
    output
  }

  override def updateGradInput(
    input: Tensor[T],
    gradOutput: Tensor[T]
  ): Tensor[T] = {

    /*
    val divGradInput1 = gradOutput.clone().cmul(weight).div(divInput2)
    val divGradInput2 = (divGradInput1.clone().div(divInput2)).cmul(divInput1)
    val squareGadO = divGradInput2.sum(divGradInput2.dim())
    val sqrtGradI = divInput2.div(sqrtInput).cmul(squareGadO)
    val sumGradI = sqrtGradI
      .div(ev.fromType(-1 * divInput1.size(divInput1.dim())))
      .expand(divInput1.size())
    val squareGradI = divInput1.cmul(sumGradI)

    val addGradO = divGradInput1.add(squareGradI)
    val addGradI = addGradO.sum(addGradO.dim())
    val negativeGradO = addGradI.sum(addGradI.dim())
    //    val negativeGradI = negativeGradO.mul(ev.fromType(-1))
    val sum2GradI = negativeGradO.div(ev.fromType(-1 * input.size(input.dim())))

    gradInput = sum2GradI.add(addGradO)
    gradInput
     */
    /** NEW CODE * */

    val divGradInput1 = gradOutput.clone().cmul(weight)
    optimzedOperation(divGradInput1, divInput2, "/")

    val divGradInput2k = divGradInput1.clone()
    optimzedOperation(divGradInput2k, divInput2, "/")
    val divGradInput2 = divGradInput2k.cmul(divInput1)

    val squareGadO = divGradInput2.sum(divGradInput2.dim())

    optimzedOperation(divInput2, sqrtInput, "/")

    val sqrtGradI = {
      optimzedOperation(divInput2, squareGadO, "*"); divInput2
    }

    val sumGradI = sqrtGradI
      .div(ev.fromType(-1 * divInput1.size(divInput1.dim())))
      .expand(divInput1.size())

    val squareGradI = divInput1.cmul(sumGradI)

    val addGradO      = divGradInput1.add(squareGradI)
    val addGradI      = addGradO.sum(addGradO.dim())
    val negativeGradO = addGradI.sum(addGradI.dim())
    val sum2GradI     = negativeGradO.div(ev.fromType(-1 * input.size(input.dim())))

    gradInput = sum2GradI.add(addGradO)
    gradInput
  }

  override def accGradParameters(
    input: Tensor[T],
    gradOutput: Tensor[T]
  ): Unit = {
    var i = 1
    gradWeight = y.clone().cmul(gradOutput)
    gradBias = gradOutput
    while (i < gradOutput.dim()) {
      gradBias = gradBias.sum(i)
      gradWeight = gradWeight.sum(i)
      i += 1
    }
    gradBias.resize(bias.size())
    gradWeight.resize(weight.size())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}

/*
class LayerNormUpdated[T: ClassTag](
    val nOutput: Int = 768,
    val eps: Double = 1e-5
)(implicit ev: TensorNumeric[T])
    extends TensorModule[T]
    with Initializable {
  val weight = Tensor.ones[T](nOutput).view(1, nOutput)
  val bias = Tensor[T](nOutput).view(1, nOutput)

  var gradWeight: Tensor[T] = Tensor[T]()
  var gradBias: Tensor[T] = Tensor[T]()

  var y: Tensor[T] = null
  var divInput1: Tensor[T] = null
  var divInput2: Tensor[T] = null
  var sqrtInput: Tensor[T] = null

  private def optimzedOperation(
      input1: Tensor[T],
      input2: Tensor[T],
      operation: String
  ) = {
    val dim = input1.dim()
    val kk = Array.fill[Int](dim - 1)(1)
    var m = 0
    var cnt = 0

    while (kk(0) < input1.size(1) + 1) {
      cnt += 1
      if (cnt < input1.dim() - 1) {
        m = 1
        while (m < kk.size) {
          kk(m) = 1
          m += 1
        }
        while (kk(1) < input1.size(2) + 1) {
          cnt += 1
          if (cnt < input1.dim() - 1) {
            m = 2
            while (m < kk.size) {
              kk(m) = 1
              m += 1
            }
            while (kk(2) < input1.size(3) + 1) {
              cnt += 1
              if (cnt < input1.dim() - 1) {} else {
                if (operation == "-") {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .sub(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else if (operation == "/") {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .div(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else if (operation == "*") {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .mul(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .add(input2.valueAt(kk(0), kk(1), kk(2), 1))
                }
              }
              kk(2) += 1
              cnt = 2
            }
          } else {
            if (operation == "-") {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .sub(input2.valueAt(kk(0), kk(1), 1))
            } else if (operation == "/") {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .div(input2.valueAt(kk(0), kk(1), 1))
            } else if (operation == "*") {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .mul(input2.valueAt(kk(0), kk(1), 1))
            } else {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .add(input2.valueAt(kk(0), kk(1), 1))
            }

          }
          kk(1) += 1
          cnt = 1
        }
      } else {
        if (operation == "-") {
          input1.narrow(1, kk(0), 1).sub(input2.valueAt(kk(0), 1))
        } else if (operation == "/") {
          input1.narrow(1, kk(0), 1).div(input2.valueAt(kk(0), 1))
        } else if (operation == "*") {
          input1.narrow(1, kk(0), 1).mul(input2.valueAt(kk(0), 1))
        } else {
          input1.narrow(1, kk(0), 1).add(input2.valueAt(kk(0), 1))
        }
      }
      kk(0) += 1
      cnt = 0
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val u = input.sum(dim).div(ev.fromType(input.size(dim)))

    divInput1 = TensorOperation.subTensor(input.clone(), u)
    val square = divInput1.clone().square()
    val s = square.sum(square.dim()).div(ev.fromType(square.size(square.dim())))
    sqrtInput = s.add(ev.fromType(eps))
    divInput2 = sqrtInput.clone().sqrt()
    y = TensorOperation.divTensor(divInput1.clone(), divInput2)
    output = y.clone().cmul(weight).add(bias)
    output
    /*val dim = input.dim()
    val u = input.sum(dim).div(ev.fromType(input.size(dim)))

    divInput1 = TensorOperation.subTensor(input.clone(), u)
    val square = divInput1.clone().square()
    val s =
      square.sum(square.dim()).div(ev.fromType(square.size(square.dim())))
    sqrtInput = s.add(ev.fromType(eps))
    divInput2 = sqrtInput.clone().sqrt()
    y = TensorOperation.divTensor(divInput1.clone(), divInput2)
    output = y.clone().cmul(weight).add(bias)
    output*/
  }

  override def updateGradInput(
      input: Tensor[T],
      gradOutput: Tensor[T]
  ): Tensor[T] = {

    val divGradInput1 = gradOutput.clone().cmul(weight).div(divInput2)
    //  below code is equal to
    //  val divGradInput2 = (divGradInput1.clone().div(divInput2))
    // .mul(ev.fromType(-1)).cmul(divInput1)
    //  val squareGadO = divGradInput2.sum(divGradInput2.dim())
    //  val sqrtGradI = divInput2.div(sqrtInput).mul(ev.fromType(0.5)).cmul(squareGadO)
    //  val sumGradI = sqrtGradI.div(ev.fromType(divInput1.size(divInput1.dim())))
    //    .expand(divInput1.size())
    //  val squareGradI = divInput1.mul(ev.fromType(2)).cmul(sumGradI)
    val divGradInput2 = (divGradInput1.clone().div(divInput2)).cmul(divInput1)
    val squareGadO = divGradInput2.sum(divGradInput2.dim())
    val sqrtGradI = divInput2.div(sqrtInput).cmul(squareGadO)
    val sumGradI = sqrtGradI
      .div(ev.fromType(-1 * divInput1.size(divInput1.dim())))
      .expand(divInput1.size())
    val squareGradI = divInput1.cmul(sumGradI)

    val addGradO = divGradInput1.add(squareGradI)
    val addGradI = addGradO.sum(addGradO.dim())
    val negativeGradO = addGradI.sum(addGradI.dim())
    //    val negativeGradI = negativeGradO.mul(ev.fromType(-1))
    val sum2GradI = negativeGradO.div(ev.fromType(-1 * input.size(input.dim())))

    gradInput = sum2GradI.add(addGradO)
    gradInput

    /** NEW CODE * */
    /*
    val divGradInput1 = gradOutput.clone().cmul(weight).div(divInput2)
    //  below code is equal to
    //  val divGradInput2 = (divGradInput1.clone().div(divInput2))
    // .mul(ev.fromType(-1)).cmul(divInput1)
    //  val squareGadO = divGradInput2.sum(divGradInput2.dim())
    //  val sqrtGradI = divInput2.div(sqrtInput).mul(ev.fromType(0.5)).cmul(squareGadO)
    //  val sumGradI = sqrtGradI.div(ev.fromType(divInput1.size(divInput1.dim())))
    //    .expand(divInput1.size())
    //  val squareGradI = divInput1.mul(ev.fromType(2)).cmul(sumGradI)

    val k0 = divGradInput1.clone()
    optimzedOperation(k0, divInput2, "/")

    val divGradInput2 = { optimzedOperation(k0, divInput1, "*"); k0 }

    val squareGadO = divGradInput2.sum(divGradInput2.dim())

    optimzedOperation(divInput2, sqrtInput, "/")
    // val sqrtGradI = divInput2.div(sqrtInput).cmul(squareGadO)
    val sqrtGradI = { optimzedOperation(divInput2, squareGadO, "*"); divInput2 }

    val sumGradI = sqrtGradI
      .div(ev.fromType(-1 * divInput1.size(divInput1.dim())))
      .expand(divInput1.size())

    val squareGradI = { optimzedOperation(divInput1, sumGradI, "*"); divInput1 }

    val addGradO = {
      optimzedOperation(divGradInput1, squareGradI, "+"); divGradInput1
    }

    val addGradI = addGradO.sum(addGradO.dim())

    val negativeGradO = addGradI.sum(addGradI.dim())
    //    val negativeGradI = negativeGradO.mul(ev.fromType(-1))

    val sum2GradI = negativeGradO.div(ev.fromType(-1 * input.size(input.dim())))

    gradInput = sum2GradI.add(addGradO)

    gradInput*/

  }

  override def accGradParameters(
      input: Tensor[T],
      gradOutput: Tensor[T]
  ): Unit = {
    var i = 1
    gradWeight = y.clone().cmul(gradOutput)
    gradBias = gradOutput
    while (i < gradOutput.dim()) {
      gradBias = gradBias.sum(i)
      gradWeight = gradWeight.sum(i)
      i += 1
    }
    gradBias.resize(bias.size())
    gradWeight.resize(weight.size())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.OUT_IN)
    Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    zeroGradParameters()
  }
}*/
