package com.intel.analytics.bigdl.dllib.keras.autograd

import com.intel.analytics.bigdl.dllib.keras.layers.KerasLayerWrapper
import com.intel.analytics.bigdl.dllib.keras.layers.internal.{InternalERF, InternalERF2}
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object ZooExtensions {
  def maskValue[T: ClassTag](bidirectional: Boolean = true, seqLen: Int)(implicit
    ev: TensorNumeric[T]
  ) = if (!bidirectional) {
    val data =
      KerasUtils.tril(Tensor.ones(seqLen, seqLen)).view(1, seqLen, seqLen)
    new Constant[T](data)
  } else null

  def metaVariableInfo[T: ClassTag](in: Variable[T]) = {
    in.node.element.getParametersTable().keySet.mkString("")
  }

  def erf[T: ClassTag](
    x: Variable[T]
  )(implicit ev: TensorNumeric[T]): Variable[T] = {
    Variable(
      new KerasLayerWrapper[T](
        new InternalERF[T]()
          .asInstanceOf[AbstractModule[Activity, Activity, T]]
      ).inputs(x.node)
    )
  }

}
