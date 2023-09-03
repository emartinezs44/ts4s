package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.internal.CustomDropout

import scala.reflect.ClassTag

/** Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each update during training time in order to prevent
  * overfitting.
  *
  * When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch
  * dimension).
  *
  * @param p
  *   Fraction of the input units to drop. Double between 0 and 1.
  * @param inputShape
  *   A Single Shape, does not include the batch dimension.
  * @tparam T
  *   Numeric type of parameter(e.g. weight, bias). Only support float/double now.
  */
class DropoutUpdated[T: ClassTag](
  override val p: Double,
  override val inputShape: Shape = null
)(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.dllib.keras.layers.internal.CustomDropout[T](
    p, inputShape
  ) with Net {

  /*
  override private[bigdl] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.param(getName()) ++
      Net.param(p, "rate")
    Net.kerasDef(this, params)
  }*/

}

object DropoutUpdated {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double,
    inputShape: Shape = null
  )(implicit ev: TensorNumeric[T]): DropoutUpdated[T] = {
    new DropoutUpdated[T](p, inputShape)
  }
}
