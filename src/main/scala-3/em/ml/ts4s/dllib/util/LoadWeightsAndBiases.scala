package em.ml.ts4s.dllib.util

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.onnx.ModelProto
import org.bytedeco.onnx.global.onnx.{ParseProtoFromBytes, check_model}
import org.json4s.*
import org.json4s.jackson.JsonMethods.*
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import com.intel.analytics.bigdl.dllib.tensor.Tensor as DTensor
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path as HPath

import java.nio.file.*
import java.io.InputStream
import java.nio.{ByteBuffer, ByteOrder}
import scala.io.Source

case class ModelNode(layer: String, dims: List[Long], capacity: Long)

object LoadWeightsAndBiases {
  implicit val formats: Formats = Serialization.formats(NoTypeHints)

  import com.github.plokhotnyuk.jsoniter_scala.macros._
  import com.github.plokhotnyuk.jsoniter_scala.core._

  given codec: JsonValueCodec[List[ModelNode]] = JsonCodecMaker.make

  def loadFromOnnx(modelPath: String, modelJson: InputStream, modelHeadJson: Option[InputStream]): Seq[(String, DTensor[Float])] = {
    val bytes             = Files.readAllBytes(Paths.get(modelPath))
    val model: ModelProto = new ModelProto

    val buffer: ByteBuffer = ByteBuffer.wrap(bytes)
    ParseProtoFromBytes(model, new BytePointer(buffer), bytes.length)
    check_model(model)

    val proto     = model.graph
    val size: Int = model.graph.initializer_size

    val json: String = {
      Source
        .fromInputStream(modelJson, "UTF-8")
        .getLines()
        .mkString("")
    }

    val parsedModel: List[ModelNode] =
      readFromString[List[ModelNode]](
        json
      )

    val headNodes = modelHeadJson match
       case Some(inputStream: InputStream) =>
         val jsonHead = Source
           .fromInputStream(inputStream, "UTF-8")
           .getLines()
           .mkString("")

         val headNodes = readFromString[List[ModelNode]](
           jsonHead
         )

         headNodes
       case None => Nil

    val totalNodes = parsedModel ++ headNodes
    (0 until size).map { i =>
       val layerName = proto.initializer(i).name().getString
       val internalBuffer: Array[Byte] =
         Array.fill(proto.initializer(i).raw_data.capacity().toInt)(0x0)
       proto.initializer(i).raw_data.get(internalBuffer)
       val iBuffer = ByteBuffer.wrap(internalBuffer)
       iBuffer.order(ByteOrder.LITTLE_ENDIAN)
       val nodeFromModel = totalNodes.find(_.layer == layerName)
       nodeFromModel match {
         case Some(n) =>
           val arrayFloat = Array.fill((n.capacity / 4).toInt)(0f)
           for (a <- 0 until n.capacity.toInt / 4) {
             arrayFloat(a) = iBuffer.getFloat
           }
           (n.layer, DTensor(arrayFloat, n.dims.toArray.map(_.toInt)))
         case _ => throw new Exception("Invalid data")
       }
    }
  }

  def loadModel(model: String, weights: String): KerasNet[Float] = {

    if (model.contains("wasb") || model.contains("hdfs")) {
      val conf              = new Configuration()
      val fs                = new HPath(model).getFileSystem(conf)
      val remotePathModel   = new HPath(model)
      val remotePathWeights = new HPath(weights)
      val localPathModel    = new HPath(s"model/${remotePathModel.getName()}")
      val localPathWeights  = new HPath(s"model/${remotePathWeights.getName()}")
      fs.copyToLocalFile(remotePathModel, localPathModel)
      fs.copyToLocalFile(remotePathWeights, localPathWeights)
      Net.load[Float](localPathModel.toString, localPathWeights.toString)
    } else {
      Net.load[Float](model, weights)
    }
  }
}
