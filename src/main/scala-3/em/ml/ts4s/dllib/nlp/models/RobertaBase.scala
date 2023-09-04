package em.ml.ts4s.dllib.nlp.models

import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.{Model, Net}
import com.intel.analytics.bigdl.dllib.keras.autograd.{Variable, DLLibExtensions}
import com.intel.analytics.bigdl.dllib.keras.layers.{Dropout, Embedding, LayerNorm}
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.utils.{MultiShape, Shape, Table}
import em.ml.ts4s.dllib.layers.Transformer

import scala.collection.mutable.ListBuffer

/** BigDL RoBERTa model implementation. This model is created using Keras Based BigDLDlib specification. Emiliano Martinez Sanchez
  */

case class RobertaInternalLayer(internalName: String, hfName: String)
case class RobertaEquivalenceGraph(nodes: List[RobertaInternalLayer])

case class RobertaBase[T: ClassTag](
  inputSeqLen: Int,
  maxPositionLen: Int = 512,
  vocab: Int,
  hiddenSize: Int,
  nBlock: Int,
  hiddenPDrop: Double,
  attnPDrop: Double,
  nHead: Int,
  initializerRange: Double = 0.02,
  bidirectional: Boolean = true,
  outputAllBlock: Boolean,
  intermediateSize: Int = 0,
  outputLastState: Boolean = false,
  useLoraInMultiHeadAttention: Boolean = false,
  var inputShape: Shape = null,
  headLayer: Option[
    (KerasLayer[Activity, Tensor[T], T], RobertaEquivalenceGraph)
  ]
)(using ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape))
  with Net {

  var graphModel: Option[RobertaEquivalenceGraph] = None

  private def buildInput(
    inputShape: Shape
  ): (Variable[T], List[Variable[T]], List[Variable[T]]) = {
    require(
      inputShape.isInstanceOf[MultiShape] &&
        inputShape.asInstanceOf[MultiShape].value.size == 3,
      "ROBERTA input must be" +
        " a list of 3 tensors (consisting of input sequence, sequence positions," +
        " attention mask)"
    )

    val input  = KerasUtils.removeBatch(inputShape).toMulti()
    val inputs = input.map(Variable(_))
    ((-inputs.last + 1.0) * (-25.0), inputs, inputs)
  }

  override def doBuild(
    inputShape: Shape
  ): AbstractModule[Activity, Activity, T] = {

    val matNamesTotal                   = ListBuffer[String]()
    val (mask, embeddingInputs, inputs) = buildInput(inputShape)
    val initPositionEmbeddingW =
      Tensor[T](maxPositionLen, hiddenSize).randn(0.0, initializerRange)

    val wordEmbeddings                         = new Embedding[Float](vocab, hiddenSize)
    val wordEmbeddingsNode                     = wordEmbeddings.from(embeddingInputs(0).squeeze(1))
    var equivalenceGraph                       = RobertaEquivalenceGraph(Nil)
    var graph: Option[RobertaEquivalenceGraph] = None
    equivalenceGraph = equivalenceGraph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(wordEmbeddingsNode),
        "embeddings.word_embeddings"
      ) :: equivalenceGraph.nodes
    )

    val positionEmbeddings = new Embedding(
      maxPositionLen,
      hiddenSize,
      initWeights = initPositionEmbeddingW
    )

    val positionEmbeddingsNode =
      positionEmbeddings.from(embeddingInputs(1).squeeze(1))
    equivalenceGraph = equivalenceGraph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(positionEmbeddingsNode),
        "embeddings.position_embeddings"
      ) :: equivalenceGraph.nodes
    )

    val embeddings = wordEmbeddingsNode + positionEmbeddingsNode
    val afterNorm  = LayerNorm(nOutput = hiddenSize, eps = 1e-5).from(embeddings)
    equivalenceGraph = equivalenceGraph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(afterNorm),
        "embeddings.LayerNorm"
      ) :: equivalenceGraph.nodes
    )

    val h = Dropout(hiddenPDrop).from(afterNorm)
    val transformer = new Transformer(
      initializerRange,
      intermediateSize,
      hiddenPDrop,
      nHead,
      bidirectional,
      inputSeqLen,
      attnPDrop,
      equivalenceGraph
    )

    val modelOutputSize = nBlock
    val modelOutput     = new Array[Variable[T]](modelOutputSize)
    val (outputZero, graph0, state, contextVector, _output, matNames) =
      transformer.block(
        h,
        hiddenSize,
        blockNumber = 0,
        inputSeqLen = null,
        attention_mask = mask,
        useLoraInMultiHeadAttention
      )
    matNamesTotal ++= matNames

    graph = Some(graph0)
    modelOutput(0) = outputZero
    for (i <- 1 until nBlock) {
      val (outputZero, graph0, _, _, _, matNames) =
        transformer.block(
          modelOutput(i - 1),
          hiddenSize,
          blockNumber = i,
          inputSeqLen = null,
          attention_mask = mask,
          useLoraInMultiHeadAttention
        )
      graph = Some(graph0)
      modelOutput(i) = outputZero
      matNamesTotal ++= matNames
    }

    graphModel = graph
    if (!headLayer.isDefined) {
      val model = Model(
        inputs.toArray,
        Array(modelOutput.last, state, contextVector, _output)
      )
      model
    } else {
      val outputWithHead = headLayer.get._1.asInstanceOf[Net].from(modelOutput.last)
      val headGraph      = headLayer.get._2
      graphModel = Some(
        RobertaEquivalenceGraph(graphModel.get.nodes ++ headGraph.nodes)
      )
      val model: Model[T] = {
        if (!outputLastState)
          Model(inputs.toArray, outputWithHead)
        else
          Model(inputs.toArray, Array(outputWithHead, modelOutput.last))
      }
      if (useLoraInMultiHeadAttention)
         model.freeze()
         // Freeze all model except low attention low projections and classification layer
         model.unFreeze((matNamesTotal += "classification_head").toSeq: _*)

      model.summary()
      model
    }
  }
}
