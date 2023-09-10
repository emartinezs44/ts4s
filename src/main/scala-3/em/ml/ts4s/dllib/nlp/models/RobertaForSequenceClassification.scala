package em.ml.ts4s.dllib.nlp.models

import scala.io.Source
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.utils.{MultiShape, Shape, Table}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.keras.{Model, Net}
import com.intel.analytics.bigdl.dllib.keras.autograd.{Variable, DLLibExtensions}
import com.intel.analytics.bigdl.dllib.keras.layers.{Activation, Dense, Dropout, Embedding, LayerNorm, Select, SoftMax}
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import em.ml.ts4s.dllib.util.{LoadWeightsAndBiases, ModelNode}
import em.ml.ts4s.dllib.nlp.models.RobertaBase
import em.ml.ts4s.dllib.extensions.*
import em.ml.ts4s.tokenizers.BpeTokenizer
import com.github.plokhotnyuk.jsoniter_scala.macros.*
import com.github.plokhotnyuk.jsoniter_scala.core.*
import em.ml.ts4s.dllib.layers.Transformer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.*
import scala3udf.Udf as udf

import scala.collection.Seq

object RobertaClassHead {
  def headShape(seqLen: Int, hiddenSize: Int) =
    Shape(seqLen, hiddenSize)

  /** Include parameters * */
  def parameters(labelsNum: Int) = Map(
    "classification.dense.weight"
      -> Array(768, 768),
    "classification.dense.bias"
      -> Array(1, 768),
    "classification.output.weight" -> Array(768, labelsNum),
    "classification.output.bias"   -> Array(1, labelsNum)
  )
}

class RobertaClassificationHead[T: ClassTag](
  hiddenSize: Int,
  drop: Double,
  labels: Int,
  multiLabel: Boolean = false,
  var inputShape: Shape = null
)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape))
  with Net {

  var graphModel: Option[RobertaEquivalenceGraph] = None
  var graph                                       = RobertaEquivalenceGraph(Nil)

  def pooler(x: Variable[T], hiddenSize: Int): Variable[T] = {
    val firstToken = Select(1, 0).from(x)
    val output     = Dense(hiddenSize).from(firstToken)

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(output),
        s"classification.dense"
      ) :: graph.nodes
    )

    Activation[Float]("tanh").from(output)
  }

  def buildInput(inputShape: Shape): Variable[T] = {
    Variable[T](inputShape)
  }

  override def doBuild(
    inputShape: Shape
  ): AbstractModule[Activity, Activity, T] = {
    val input = buildInput(inputShape)
    val a     = Dropout(drop).from(input)
    val b     = pooler(a, hiddenSize)
    val c     = Dropout(drop).from(b)
    val d = {
      if (!multiLabel)
        Dense(labels, activation = "log_softmax").from(c)
      else Dense(labels /*, activation = "sigmoid"*/ ).from(c)
    }

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(d),
        s"classification.output"
      ) :: graph.nodes
    )

    graphModel = Some(graph)
    Model(input, d).setName("classification_head")
  }
}

class RobertaForSequenceClassification(
  seqLen: Int,
  vocab: Int = 50262,
  hiddenSize: Int,
  nBlock: Int = 12,
  nHead: Int = 12,
  intermediateSize: Int = 3072,
  hiddenPDrop: Float = 0.1,
  attnPDrop: Float = 0.1,
  outputAllBlock: Boolean = false,
  useLoraInMultiHeadAtt: Boolean = false
) extends DlModel {

  val robertaBaseShape = MultiShape(
    List(
      Shape(1, seqLen),
      Shape(1, seqLen),
      Shape(1, 1, 1, seqLen)
    )
  )

  def convertModelFromOnnx(
    onnxModelPath: String,
    outputClasses: Int,
    modelStorePath: String,
    weightsStorePath: String,
    multiLabel: Boolean = false,
    printSummary: Boolean = false
  ): AbstractModule[Activity, Activity, Float] = {
    val jsonStreamEncoderPath = getClass.getClassLoader.getResourceAsStream("models/roberta_base.json")
    val jsonStreamHeadPath    = getClass.getClassLoader.getResourceAsStream("models/classification_head.json")
    val tensorsList           = LoadWeightsAndBiases.loadFromOnnx(onnxModelPath, jsonStreamEncoderPath, Some(jsonStreamHeadPath))

    val model = new RobertaBase[Float](
      vocab = 50262,
      hiddenSize = hiddenSize,
      nBlock = 12,
      nHead = 12,
      intermediateSize = 3072,
      hiddenPDrop = 0.1,
      attnPDrop = 0.1,
      maxPositionLen = seqLen,
      outputAllBlock = false,
      inputSeqLen = seqLen,
      headLayer = None,
      useLoraInMultiHeadAttention = useLoraInMultiHeadAtt
    )

    val head = new RobertaClassificationHead[Float](
      hiddenSize = model.hiddenSize,
      drop = model.hiddenPDrop,
      labels = outputClasses,
      multiLabel = multiLabel
    )

    val modelWithHead = model.copy[Float](
      headLayer = Some(
        (head.doBuild(RobertaClassHead.headShape(seqLen, hiddenSize)).asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], head.graphModel.get)
      )
    )

    val modelBuilt: AbstractModule[Activity, Activity, Float] =
      modelWithHead.doBuild(robertaBaseShape)

    val p0: Seq[(Seq[Int], String)] = modelBuilt
      .asInstanceOf[Model[Float]]
      .parametersWithName()
      .map { case (tensor, name) => (tensor.size(), name) }
      .toSeq
      .map { case (dims, name) => (dims.toSeq, name) }

    val p1: Array[List[Int]] = modelBuilt.parameters()._1.map(_.size().toList)
    require(modelWithHead.graphModel.isDefined, "Graph Not Defined")
    val parametersInitialized: Seq[(String, Tensor[Float])] = p0.map { case (_, name) =>
      var isBias = false
      val layerName = if (name.startsWith("bias_")) {
        isBias = true
        name.substring("bias_".length)
      } else name
      
      val node = modelWithHead.graphModel.get.nodes.find(_.internalName == layerName)
      if (node.isDefined) {
        val hfName = node.get.hfName
        val nameToSearch =
          if (isBias)
            s"$hfName.bias"
          else
            s"$hfName.weight"

        val result = tensorsList.find(_._1 == nameToSearch)
        if (result.isDefined) {
          (result.get._1, result.get._2)
        } else {
          val headNodes = RobertaClassHead.parameters(labelsNum = outputClasses)
          val headNode  = headNodes.get(nameToSearch)
          val loraNodes = Transformer.loraParameters(blocks = model.nBlock, hiddenSize = model.hiddenSize)
          if (headNode.isDefined) {
            (nameToSearch, Tensor[Float](headNode.get).rand())
          } else {
            (nameToSearch, Tensor[Float](loraNodes(nameToSearch)).rand())
          }
        }
      } else {
        throw new Exception("Node not found")
      }
    }

    (p1 zip parametersInitialized.map(_._2)).foreach { case (dim, tensor) =>
      val diff = dim.length - tensor.size().length
      if (diff > 0) {
        (0 to diff - 1).foreach(_ => tensor.addSingletonDimension())
      } else if (diff < 0) {
        tensor.squeeze()
      }
    }

    if (printSummary)
      modelBuilt.asInstanceOf[KerasNet[Float]].summary()

    modelBuilt.setWeightsBias(parametersInitialized.map(_._2).toArray)
    modelBuilt.saveModule(modelStorePath, weightsStorePath)
  }

  override def unfreeze: List[String] = {
    "classification_head" :: (0 until nBlock).flatMap { block =>
      Array(s"query_${block}.a", s"value_${block}.a", s"query_${block}.b", s"value_${block}.b")
    }.toList
  }
}

object RobertaForSequenceClassification {
  import scala3encoders.encoder

  private val vocabString = Source.fromInputStream(
    getClass.getClassLoader.getResourceAsStream("tokenizers/bpe/vocab.json")
  )

  private val mergesString = Source.fromInputStream(
    getClass.getClassLoader.getResourceAsStream("tokenizers/bpe/merges.txt")
  )

  private val lines = mergesString.getLines()

  private val mapMerges =
    lines.map(_.split(" ")).map(el => el(0) -> el(1)).zipWithIndex

  given codec: JsonValueCodec[Map[String, Int]] = JsonCodecMaker.make(CodecMakerConfig.withMapMaxInsertNumber(53000))

  private val mapVocab =
    readFromString(vocabString.mkString(""))

  @transient lazy val tokenizer = BpeTokenizer.forModel(
    "roberta",
    merges = mapMerges.toMap,
    vocab = mapVocab,
    padWithSentenceTokens = false
  )

  def tokenizeStr(str: String) = {
    val tokens = tokenizer.tokenizeSubText(str, 0)
    val t = tokens.map(el =>
      el.copy(
        token = new String(
          el.token
            .getBytes("utf-8")
            .map(java.lang.Byte.toUnsignedInt(_))
            .map(_.toChar)
        )
      )
    )
    tokenizer.encode(t).map(_.pieceId).map(_ + 1)
  }

  def tokenizeDataframeColumn(data: DataFrame)(inputColumnName: String, outputColumnName: String): DataFrame =
     val tokenizeUdf = udf((str: String) => tokenizeStr(str))
     data.withColumn(outputColumnName, tokenizeUdf(col(inputColumnName)))
}
