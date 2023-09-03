package em.ml.ts4s.dllib.layers

import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.autograd.ZooExtensions.maskValue
import com.intel.analytics.bigdl.dllib.keras.autograd.{AutoGrad, Variable, ZooExtensions}
import com.intel.analytics.bigdl.dllib.keras.layers.{Activation, Dense, Dropout, LayerNorm, Permute, Reshape, Select, SoftMax}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import em.ml.ts4s.dllib.nlp.models.{RobertaEquivalenceGraph, RobertaInternalLayer}

import scala.reflect.ClassTag

class Transformer[T: ClassTag](
  val initializerRange: Double = 0.02,
  val intermediateSize: Int = 0,
  val hiddenPDrop: Double,
  val nHead: Int,
  val bidirectional: Boolean = true,
  val seqLen: Int = 0,
  val attnPDrop: Double,
  var graph: RobertaEquivalenceGraph
)(implicit ev: TensorNumeric[T]) {

  def projectionLayer(outputSize: Int, name: String): Net = {
    val gName = s"${name}_${java.util.UUID.randomUUID()}"
    new Dense[T](outputSize).setName(gName)
  }

  /*def gelu(x: Variable[T]): Variable[T] = {
    x * 0.5 * (Activation("tanh").from((AutoGrad.square(x) * x * 0.044715 + x)
   * (scala.math.sqrt(2 / scala.math.Pi))) + 1)
  }*/

  def gelu(x: Variable[T]): Variable[T] = {
    val y = x / math.sqrt(2.0)
    val e = ZooExtensions.erf(y)
    x * 0.5 * (e + 1.0)
  }

  def splitHeads(
    x: Variable[T],
    nHead: Int,
    k: Boolean = false
  ): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray

    val newSizes =
      sizes.drop(1).dropRight(1) ++ Array(nHead, sizes.last / nHead)
    val r = Reshape(newSizes).from(x)

    /** This operation is before the attention scores operation * */
    Permute(Array(2, 1, 3)).from(r)
  }

  def mergeHeads(x: Variable[T]): Variable[T] = {
    val p     = AutoGrad.contiguous(Permute[T](Array(2, 1, 3)).from(x))
    val sizes = p.getOutputShape().toSingle().toArray
    Reshape(
      sizes.drop(1).dropRight(2) ++ Array(sizes.last * sizes(sizes.length - 2))
    ).from(p)
  }

  def mlp(
    x: Variable[T],
    hiddenSize: Int,
    name: String,
    blockNumber: Int
  ): (Variable[T], Variable[T]) = {
    val size = if (intermediateSize > 0) intermediateSize else hiddenSize * 4
    val h    = projectionLayer(size, name).from(x)

    graph = graph.copy(
      RobertaInternalLayer(
        ZooExtensions.metaVariableInfo(h),
        s"encoder.layer.${blockNumber}.intermediate.dense"
      ) :: graph.nodes
    )

    val a  = gelu(h)
    val h2 = projectionLayer(hiddenSize, name).from(a)

    graph = graph.copy(
      RobertaInternalLayer(
        ZooExtensions.metaVariableInfo(h2),
        s"encoder.layer.${blockNumber}.output.dense"
      ) :: graph.nodes
    )

    /** Test without dropout  * */
    (Dropout(hiddenPDrop).from(h2), h2)
  }

  def attn(
    q: Variable[T],
    k: Variable[T],
    v: Variable[T],
    scale: Boolean = false,
    inputSeqLen: Variable[T],
    attention_mask: Variable[T] = null
  ): (Variable[T], Variable[T]) = {

    val maskValueInstance = maskValue(bidirectional, seqLen)
    var w                 = AutoGrad.mm(q, Permute(Array(1, 3, 2)).from(k))
    if (scale)
      w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)

    if (!bidirectional) {
      w = w * maskValueInstance + (maskValueInstance * (-1) + 1) * -1e9
    }

    /** This operation takes time * */
    if (attention_mask != null) {
      w = w + AutoGrad
        .contiguous(
          attention_mask
            .expand(attention_mask.getOutputShape().toSingle())
        )
        .expand(attention_mask.getOutputShape().toSingle())
    }

    w = SoftMax().from(w)

    /** Test without dropout * */
    val w1 = Dropout[T](attnPDrop).from(w)
    val w2 = AutoGrad.mm(w1, v)

    (w2, w2)
  }

  def multiHeadSelfAttention(
    x: Variable[T],
    hiddenSize: Int,
    inputSeqLen: Variable[T],
    block: Int,
    attention_mask: Variable[T] = null
  ): (Variable[T], Variable[T], Variable[T], Variable[T]) = {

    val query = projectionLayer(hiddenSize, s"Query_$block").from(x)
    val key   = projectionLayer(hiddenSize, s"Key_$block").from(x)
    val value = projectionLayer(hiddenSize, s"Value_$block").from(x)

    graph = graph.copy(
      RobertaInternalLayer(
        ZooExtensions.metaVariableInfo(query),
        s"encoder.layer.${block}.attention.self.query"
      )
        :: RobertaInternalLayer(
          ZooExtensions.metaVariableInfo(key),
          s"encoder.layer.${block}.attention.self.key"
        )
        :: RobertaInternalLayer(
          ZooExtensions.metaVariableInfo(value),
          s"encoder.layer.${block}.attention.self.value"
        )
        :: graph.nodes
    )

    val q = splitHeads(query, nHead)
    val k = splitHeads(key, nHead, k = true)
    val v = splitHeads(value, nHead)

    val (a, context_layer) = attn(q, k, v, true, inputSeqLen = inputSeqLen, attention_mask = attention_mask)
    val m                  = mergeHeads(a)
    val n                  = projectionLayer(hiddenSize, s"AfterMergeDense_${block}").from(m)

    graph = graph.copy(
      RobertaInternalLayer(
        ZooExtensions.metaVariableInfo(n),
        s"encoder.layer.${block}.attention.output.dense"
      ) :: graph.nodes
    )

    // State -> Context Vector -> Output dense
    /** Test without  dropout * */
    (Dropout(hiddenPDrop).from(n), x, a, context_layer)
  }

  def block(
    x: Variable[T],
    hiddenSize: Int,
    inputSeqLen: Variable[T],
    epsilon: Double = 1e-5,
    blockNumber: Int,
    attention_mask: Variable[T] = null
  ): (Variable[T], RobertaEquivalenceGraph, Variable[T], Variable[T], Variable[T]) = {

    val (a, state, contextVector, context_layer) = multiHeadSelfAttention(
      x,
      hiddenSize,
      inputSeqLen = inputSeqLen,
      blockNumber,
      attention_mask = attention_mask
    )

    val n = LayerNorm(hiddenSize, epsilon).from(x + a)

    graph = graph.copy(
      RobertaInternalLayer(
        ZooExtensions.metaVariableInfo(n),
        s"encoder.layer.${blockNumber}.attention.output.LayerNorm"
      ) :: graph.nodes
    )

    val (m, _) =
      mlp(n, hiddenSize, s"AfterLayerNormInBlock_${blockNumber}", blockNumber)
    val h = LayerNorm(hiddenSize, epsilon).from(n + m)

    graph = graph.copy(
      RobertaInternalLayer(
        ZooExtensions.metaVariableInfo(h),
        s"encoder.layer.${blockNumber}.output.LayerNorm"
      ) :: graph.nodes
    )

    (h, graph, state, contextVector, context_layer)
  }
}
