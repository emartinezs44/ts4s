package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.nn.{LayerNormUpdated as BDLayerNP}

import scala.reflect.ClassTag

class LayerNormUpdated[T: ClassTag](
  val nOutput: Int = 768,
  val eps: Double = 1e-5,
  val inputShape: Shape = null
)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
  with Net {

  override def doBuild(
    inputShape: Shape
  ): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = new BDLayerNP[T](nOutput, eps)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    Shape(input.slice(0, input.length - 1) ++ Array(nOutput))
  }
}

object LayerNormUpdated {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int = 768,
    eps: Double = 1e-5,
    inputShape: Shape = null
  )(implicit ev: TensorNumeric[T]): LayerNormUpdated[T] = {
    new LayerNormUpdated[T](nOutput, eps, inputShape)
  }
}
