package em.ml.ts4s.dllib.extensions

import com.intel.analytics.bigdl.dllib.keras.Model
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Table

import scala.collection.mutable.ArrayBuffer

type NNParams = (Array[Tensor[Float]], Array[Tensor[Float]])

extension (mod: Model[Float])
  def parametersWithName() = {
    val weightsStorage = ArrayBuffer[Option[(Tensor[Float], String)]]()
    var weights        = ArrayBuffer[(Tensor[Float], String)]()
    val gradient       = ArrayBuffer[Tensor[Float]]()
    mod.modules.foreach(m => {
      val params: NNParams = m.parameters()
      val name: Table      = m.getParametersTable()
      name.foreach { case (a, b) =>
        weightsStorage += {
          b.asInstanceOf[Table].get[Tensor[Float]]("weight") match {
            case Some(value) =>
              Some(value, a.asInstanceOf[String])
            case _ => None
          }
        }
      }

      val mapping = weightsStorage.filter(_.isDefined).map(_.get).toMap
      val (w, _)  = params
      weights = w.foldLeft(ArrayBuffer[(Tensor[Float], String)]())((acc, b) => {
        mapping.get(b) match {
          case Some(_) =>
            val layerName = mapping(b)
            acc.+:(b, layerName)
          case _ => acc.+:(b, s"bias_${acc.head._2}")
        }
      })
    })

    weights.reverse.toArray
  }
