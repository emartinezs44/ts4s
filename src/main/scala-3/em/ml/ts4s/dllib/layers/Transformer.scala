package em.ml.ts4s.dllib.layers

import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.autograd.DLLibExtensions.maskValue
import com.intel.analytics.bigdl.dllib.keras.autograd.{AutoGrad, Variable, DLLibExtensions}
import com.intel.analytics.bigdl.dllib.keras.layers.{Activation, Dense, Dropout, LayerNorm, Permute, Reshape, Select, SoftMax}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import em.ml.ts4s.dllib.nlp.models.{RobertaEquivalenceGraph, RobertaInternalLayer}

import scala.collection.mutable.ListBuffer
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

  private def projectionLayer(outputSize: Int, name: String): (Net, String) = {
    val gName = s"${name}_${java.util.UUID.randomUUID()}"
    (Dense[T](outputSize).setName(gName), gName)
  }

  def gelu(x: Variable[T]): Variable[T] = {
    val y = x / math.sqrt(2.0)
    val e = DLLibExtensions.erf(y)
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
    val h    = projectionLayer(size, name)._1.from(x)

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(h),
        s"encoder.layer.${blockNumber}.intermediate.dense"
      ) :: graph.nodes
    )

    val a  = gelu(h)
    val h2 = projectionLayer(hiddenSize, name)._1.from(a)

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(h2),
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

  def addLowRankMatrices(rank: Int = 8, alpha: Int = 32, output: Int, input: Variable[T], name: String, stdName: String) = {
    val scale = alpha / rank
    val a     = Dense[T](outputDim = rank, bias = false, init = "normal").setName(s"${name}.a").from(input)
    val b     = Dense[T](outputDim = output, init = "zero").setName(s"${name}.b").from(a)
    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(a),
        s"${stdName}.a"
      )
        :: RobertaInternalLayer(
          DLLibExtensions.metaVariableInfo(b),
          s"${stdName}.b"
        )
        :: graph.nodes
    )
    b * scale
  }

  def multiHeadSelfAttention(
    x: Variable[T],
    hiddenSize: Int,
    inputSeqLen: Variable[T],
    block: Int,
    attention_mask: Variable[T] = null,
    useLoraInMultiHeadAttention: Boolean = false
  ): (Variable[T], Variable[T], Variable[T], Variable[T], ListBuffer[String]) = {

    val matNames = ListBuffer[String]()
    val (query, key, value) = {
      if (!useLoraInMultiHeadAttention)
         val q = projectionLayer(hiddenSize, s"Query_$block")._1.from(x)
         val k = projectionLayer(hiddenSize, s"Key_$block")._1.from(x)
         val v = projectionLayer(hiddenSize, s"Value_$block")._1.from(x)
         graph = graph.copy(
           RobertaInternalLayer(
             DLLibExtensions.metaVariableInfo(q),
             s"encoder.layer.${block}.attention.self.query"
           )
             :: RobertaInternalLayer(
               DLLibExtensions.metaVariableInfo(k),
               s"encoder.layer.${block}.attention.self.key"
             )
             :: RobertaInternalLayer(
               DLLibExtensions.metaVariableInfo(v),
               s"encoder.layer.${block}.attention.self.value"
             )
             :: graph.nodes
         )
         (q, k, v)
      else
         val q = {
           val (node, _) = projectionLayer(hiddenSize, s"query_$block")
           node.from(x)
         }
         val qLora =
           q + addLowRankMatrices(output = hiddenSize, input = q, name = s"query_$block", stdName = s"encoder.layer.${block}.attention.self.query")
         matNames += s"query_$block.a"
         matNames += s"query_$block.b"

         val k = projectionLayer(hiddenSize, s"Key_$block")._1.from(x)
         val v = {
           val (node, _) = projectionLayer(hiddenSize, s"Value_$block")
           node.from(x)
         }
         val vLora =
           v + addLowRankMatrices(output = hiddenSize, input = v, name = s"value_$block", stdName = s"encoder.layer.${block}.attention.self.value")
         matNames += s"value_$block.a"
         matNames += s"value_$block.b"

         graph = graph.copy(
           RobertaInternalLayer(
             DLLibExtensions.metaVariableInfo(q),
             s"encoder.layer.${block}.attention.self.query"
           )
             :: RobertaInternalLayer(
               DLLibExtensions.metaVariableInfo(k),
               s"encoder.layer.${block}.attention.self.key"
             )
             :: RobertaInternalLayer(
               DLLibExtensions.metaVariableInfo(v),
               s"encoder.layer.${block}.attention.self.value"
             )
             :: graph.nodes
         )
         (qLora, k, vLora)
    }

    val q = splitHeads(query, nHead)
    val k = splitHeads(key, nHead, k = true)
    val v = splitHeads(value, nHead)

    val (a, context_layer) = attn(q, k, v, true, inputSeqLen = inputSeqLen, attention_mask = attention_mask)
    val m                  = mergeHeads(a)
    val n                  = projectionLayer(hiddenSize, s"AfterMergeDense_${block}")._1.from(m)

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(n),
        s"encoder.layer.${block}.attention.output.dense"
      ) :: graph.nodes
    )

    // State -> Context Vector -> Output dense
    /** Test without  dropout * */
    (Dropout(hiddenPDrop).from(n), x, a, context_layer, matNames)
  }

  def block(
    x: Variable[T],
    hiddenSize: Int,
    inputSeqLen: Variable[T],
    epsilon: Double = 1e-5,
    blockNumber: Int,
    attention_mask: Variable[T] = null,
    useLoraInMultiHeadAttention: Boolean = false
  ): (Variable[T], RobertaEquivalenceGraph, Variable[T], Variable[T], Variable[T], ListBuffer[String]) = {

    val (a, state, contextVector, context_layer, matNames) = multiHeadSelfAttention(
      x,
      hiddenSize,
      inputSeqLen = inputSeqLen,
      blockNumber,
      attention_mask = attention_mask,
      useLoraInMultiHeadAttention
    )

    val n = LayerNorm(hiddenSize, epsilon).from(x + a)

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(n),
        s"encoder.layer.${blockNumber}.attention.output.LayerNorm"
      ) :: graph.nodes
    )

    val (m, _) =
      mlp(n, hiddenSize, s"Intermediate_${blockNumber}", blockNumber)

    val h = LayerNorm(hiddenSize, epsilon).from(n + m)

    graph = graph.copy(
      RobertaInternalLayer(
        DLLibExtensions.metaVariableInfo(h),
        s"encoder.layer.${blockNumber}.output.LayerNorm"
      ) :: graph.nodes
    )

    (h, graph, state, contextVector, context_layer, matNames)
  }
}

object Transformer {
  def loraParameters(rank: Int = 8, blocks: Int, hiddenSize: Int) =
     val nodes = 0 until blocks map { i =>
       Array(
         (s"encoder.layer.${i}.attention.self.query.a.weight" -> Array(hiddenSize, rank)),
         (s"encoder.layer.${i}.attention.self.query.a.bias"   -> Array(1, rank)),
         (s"encoder.layer.${i}.attention.self.query.b.weight" -> Array(rank, hiddenSize)),
         (s"encoder.layer.${i}.attention.self.query.b.bias"   -> Array(1, hiddenSize)),
         (s"encoder.layer.${i}.attention.self.key.a.weight"   -> Array(hiddenSize, rank)),
         (s"encoder.layer.${i}.attention.self.key.a.bias"     -> Array(1, rank)),
         (s"encoder.layer.${i}.attention.self.key.b.weight"   -> Array(rank, hiddenSize)),
         (s"encoder.layer.${i}.attention.self.key.b.bias"     -> Array(1, hiddenSize)),
         (s"encoder.layer.${i}.attention.self.value.a.weight" -> Array(hiddenSize, rank)),
         (s"encoder.layer.${i}.attention.self.value.a.bias"   -> Array(1, rank)),
         (s"encoder.layer.${i}.attention.self.value.b.weight" -> Array(rank, hiddenSize)),
         (s"encoder.layer.${i}.attention.self.value.b.bias"   -> Array(1, hiddenSize))
       )
     }
     nodes.flatten.toMap
}
