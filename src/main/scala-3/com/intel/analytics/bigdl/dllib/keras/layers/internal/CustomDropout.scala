package com.intel.analytics.bigdl.dllib.keras.layers.internal

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity, IdentityOutputShape}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape

import scala.reflect.ClassTag

/** Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each update during training time in order to prevent
  * overfitting.
  *
  * When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch
  * dimension).
  *
  * @param p
  *   Fraction of the input units to drop. Double between 0 and 1.
  * @tparam T
  *   Numeric type of parameter(e.g. weight, bias). Only support float/double now.
  */
class CustomDropout[T: ClassTag](val p: Double, val inputShape: Shape = null)(implicit
  ev: TensorNumeric[T]
) extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape))
  with IdentityOutputShape {

  override def doBuild(
    inputShape: Shape
  ): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.dllib.nn.Dropout2(p)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Dropout {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double,
    inputShape: Shape = null
  )(implicit ev: TensorNumeric[T]): CustomDropout[T] = {
    new CustomDropout[T](p, inputShape)
  }
}
