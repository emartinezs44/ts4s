/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.nn.ExtraTensorOps.extraTensorOps
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.dllib.tensor.{DenseTensorApply, DoubleType, FloatType, Storage, Tensor, TensorFunc2}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import com.intel.analytics.bigdl.dllib.utils.Shape

import java.util.concurrent.ThreadLocalRandom
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

object ExtraTensorOps {

  implicit class extraTensorOps[T: ClassTag](in: Tensor[T])(implicit
    ev: TensorNumeric[T]
  ) {
    def bernoulli2(p: Double): Tensor[T] = {
      if (in.isContiguous()) {
        var i      = 0
        val total  = in.nElement()
        val data   = in.storage().array()
        val offset = in.storageOffset() - 1
        in.getType() match {
          case FloatType =>
            val floatData = data.asInstanceOf[Array[Float]]
            while (i < total) {
              floatData(offset + i) = if (ThreadLocalRandom.current().nextFloat() <= p) {
                1
              } else {
                0
              }
              i += 1
            }
          case DoubleType =>
            val doubleData = data.asInstanceOf[Array[Double]]
            while (i < total) {
              doubleData(offset + i) = if (ThreadLocalRandom.current().nextFloat() <= p) {
                1
              } else {
                0
              }
              i += 1
            }
          case _ =>
            while (i < total) {
              data(offset + i) = if (RNG.bernoulli(p)) {
                ev.fromType[Int](1)
              } else {
                ev.fromType[Int](0)
              }
              i += 1
            }
        }
      } else {
        val func = new TensorFunc2[T] {
          override def apply(data: Array[T], index: Int): Unit = {
            data(index) = if (RNG.bernoulli(p)) {
              ev.fromType[Int](1)
            } else {
              ev.fromType[Int](0)
            }
          }
        }
        DenseTensorApply.apply1[T](in, func)
      }
      in
    }

  }
}

/** Dropout masks(set to zero) parts of input using a bernoulli distribution. Each input element has a probability initP of being dropped. If `scale`
  * is true(true by default), the outputs are scaled by a factor of `1/(1-initP)` during training. During evaluating, output is the same as input.
  *
  * It has been proven an effective approach for regularization and preventing co-adaptation of feature detectors. For more details, plese see
  * [Improving neural networks by preventing co-adaptation of feature detectors] (https://arxiv.org/abs/1207.0580)
  *
  * @param initP
  *   the probability p
  * @param inplace
  *   whether to make `input` and `output` share the same storage
  * @param scale
  *   whether to scale the output by a factor of `1 / (1 - p)`
  */
@SerialVersionUID(-4636332259181125718L)
class Dropout2[T: ClassTag](
  val initP: Double = 0.5,
  val inplace: Boolean = false,
  var scale: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  private var p    = initP
  var noise        = Tensor[T]()
  var isResampling = true

  @transient
  protected var results: Array[Future[Unit]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inplace) {
      this.output = input
    } else {
      this.output.resizeAs(input).copy(input)
    }

    /*
    if (results == null) {
      results = new Array[Future[Unit]](Engine.model.getPoolSize)
    }*/
    if (train) {
      noise.resizeAs(input)
      if (isResampling) {
        noise.bernoulli2(1 - p)

        if (scale) {
          noise.div(ev.fromType[Double](1 - p))
        }
      }

      this.output.cmul(noise)
    } else if (!scale) {
      this.output.mul(ev.fromType[Double](1 - p))
    } else {
      output
    }
  }

  override def updateGradInput(
    input: Tensor[T],
    gradOutput: Tensor[T]
  ): Tensor[T] = {
    if (results == null) {
      results = new Array[Future[Unit]](Engine.model.getPoolSize)
    }
    if (train) {
      if (inplace) {
        this.gradInput = gradOutput
      } else {
        this.gradInput.resizeAs(gradOutput).copy(gradOutput)
      }

      if (gradInput.isContiguous()) {
        val noiseData       = noise.storage().array()
        var taskSize        = noise.nElement() / Engine.model.getPoolSize
        var extraTask       = noise.nElement() % Engine.model.getPoolSize
        val gradInputData   = gradInput.storage().array()
        val gradInputOffset = gradInput.storageOffset() - 1
        var allocated       = 0
        var i               = 0
        while (allocated < noise.nElement()) {
          val start = allocated
          allocated += taskSize
          if (extraTask > 0) {
            allocated += 1
            extraTask -= 1
          }
          val end = allocated
          results(i) = Engine.model.invoke(() => {
            var k = start
            while (k < end) {
              gradInputData(gradInputOffset + k) = ev.times(gradInputData(gradInputOffset + k), noiseData(k))
              k += 1
            }
          })
          i += 1
        }

        Engine.model.sync(results)

        this.gradInput
      } else {
        this.gradInput.cmul(noise)
      }
    } else {
      throw new IllegalArgumentException("backprop only defined while training")
    }

    this.gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    noise.set()
    this
  }

  /** Set current probability to be dropped.
    * @param p
    *   new probability
    * @return
    */
  def setP(p: Double): this.type = {
    this.p = p
    this
  }

  override def toString(): String = {
    s"${getPrintName}($p)"
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object Dropout2 {
  def apply[T: ClassTag](
    initP: Double = 0.5,
    inplace: Boolean = false,
    scale: Boolean = true
  )(implicit ev: TensorNumeric[T]): Dropout2[T] = {
    new Dropout2[T](initP, inplace, scale)
  }
}
