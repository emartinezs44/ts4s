package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.nn.SoftMax2
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, IdentityOutputShape}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape

import scala.reflect.ClassTag

class SoftMaxUpdated[T: ClassTag](val inputShape: Shape = null)(implicit
  ev: TensorNumeric[T]
) extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
  with IdentityOutputShape
  with Net {

  override def doBuild(
    inputShape: Shape
  ): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = new SoftMax2()
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SoftMaxUpdated {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputShape: Shape = null
  )(implicit ev: TensorNumeric[T]): SoftMaxUpdated[T] = {
    new SoftMaxUpdated[T](inputShape)
  }
}
