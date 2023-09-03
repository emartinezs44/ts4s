package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/** Applies the SoftMax function to an n-dimensional input Tensor, rescaling them so that the elements of the n-dimensional output Tensor lie in the
  * range (0, 1) and sum to 1. Softmax is defined as: f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift) where shift = max_i(x_i). Currently only
  * support apply softmax normalization to the last dim.
  */
class SoftMax2[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim        = input.dim()
    val sizes      = input.size()
    val shift      = input.max(dim)._1
    val shiftInput = input.clone()

    if (dim <= 4 && dim > 1) {
      optimzedOperation(shiftInput, shift, "-")
    } else {
      shiftInput.sub(shift.expand(sizes).contiguous())
    }

    val exp = shiftInput.exp()

    val clonedExp = exp.clone()
    val sum       = clonedExp.sum(dim)

    if (dim <= 4 && dim > 1) {
      optimzedOperation(clonedExp, sum, "/")
    } else {
      clonedExp.div(sum.expand(sizes).contiguous())
    }
    output = clonedExp

    output
  }

  override def updateGradInput(
    input: Tensor[T],
    gradOutput: Tensor[T]
  ): Tensor[T] = {

    /*
    val dim = input.dim()
    val sum = (output.clone().cmul(gradOutput)).sum(dim)
    gradInput = output.clone().cmul(gradOutput - sum.expand(input.size()))
    gradInput
     */

    val dim           = input.dim()
    val sum           = (output.clone().cmul(gradOutput)).sum(dim)
    val gradOutputAux = gradOutput.clone()
    optimzedOperation(gradOutputAux, sum.expand(input.size()), "-")
    gradInput = output.clone().cmul(gradOutputAux)
    gradInput
  }

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
                } else {
                  input1
                    .narrow(1, kk(0), 1)
                    .narrow(2, kk(1), 1)
                    .narrow(3, kk(2), 1)
                    .div(input2.valueAt(kk(0), kk(1), kk(2), 1))
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
            } else {
              input1
                .narrow(1, kk(0), 1)
                .narrow(2, kk(1), 1)
                .div(input2.valueAt(kk(0), kk(1), 1))
            }

          }
          kk(1) += 1
          cnt = 1
        }
      } else {
        if (operation == "-") {
          input1.narrow(1, kk(0), 1).sub(input2.valueAt(kk(0), 1))
        } else {
          input1.narrow(1, kk(0), 1).div(input2.valueAt(kk(0), 1))
        }
      }
      kk(0) += 1
      cnt = 0
    }
  }
}
