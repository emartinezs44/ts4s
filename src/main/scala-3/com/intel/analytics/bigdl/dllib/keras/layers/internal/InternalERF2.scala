package com.intel.analytics.bigdl.dllib.keras.layers.internal

import com.intel.analytics.bigdl.dllib.common.zooMKLBlas2
import com.intel.analytics.bigdl.dllib.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[bigdl] class InternalERF2[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val derivativeFactor = ev.fromType(1.1283791670955126)
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    zooMKLBlas2.erf(output)
    output
  }

  override def updateGradInput(
    input: Tensor[T],
    gradOutput: Tensor[T]
  ): Tensor[T] = {
    val tensor     = Tensor().resizeAs(input).copy(input)
    val derivative = (-tensor.pow(ev.fromType(2))).exp().mul(derivativeFactor)
    gradInput = gradOutput.cmul(derivative)
    gradInput
  }
}
