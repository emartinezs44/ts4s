package em.ml.ts4s.tokenizers

/* Adapted from sparknlp project https://github.com/JohnSnowLabs/spark-nlp */

import scala.collection.{Map, Seq}

case class IndexedToken(token: String, begin: Int = 0, end: Int = 0)

case class TokenPiece(wordpiece: String, token: String, pieceId: Int, isWordStart: Boolean, begin: Int, end: Int)

case class Sentence(content: String, start: Int, end: Int, index: Int, metadata: Option[Map[String, String]] = None)

object Sentence {
  def fromTexts(texts: String*): Seq[Sentence] = {
    var idx = 0
    texts.zipWithIndex.map { case (text, textIndex) =>
      val sentence = Sentence(text, idx, idx + text.length - 1, textIndex)
      idx += text.length + 1
      sentence
    }
  }
}

private[ts4s] class SpecialTokens(
  vocab: Map[String, Int],
  startTokenString: String,
  endTokenString: String,
  unkTokenString: String,
  maskTokenString: String,
  padTokenString: String,
  additionalStrings: Array[String] = Array()
) {
  val allTokenStrings: Array[String] = Array(maskTokenString, startTokenString, endTokenString, unkTokenString, padTokenString) ++ additionalStrings
  for (specialTok <- allTokenStrings)
    require(vocab.contains(specialTok), s"Special Token '$specialTok' needs to be in vocabulary.")

  val sentenceStart: SpecialToken = SpecialToken(startTokenString, vocab(startTokenString))
  val sentenceEnd: SpecialToken   = SpecialToken(endTokenString, vocab(endTokenString))
  val unk: SpecialToken           = SpecialToken(unkTokenString, vocab(unkTokenString))
  val mask: SpecialToken = SpecialToken(
    maskTokenString,
    vocab(maskTokenString),
    lstrip = true
  )
  val pad: SpecialToken = SpecialToken(padTokenString, vocab(padTokenString))

  val additionalTokens: Array[SpecialToken] =
    additionalStrings.map((tok: String) => SpecialToken(tok, vocab(tok)))

  val allTokens: Set[SpecialToken] =
    Set(mask, sentenceStart, sentenceEnd, unk, pad) ++ additionalTokens

  def contains(s: String): Boolean = allTokens.contains(SpecialToken(content = s, id = 0))
}

private[ts4s] object SpecialTokens {
  def getSpecialTokensForModel(modelType: String, vocab: Map[String, Int]): SpecialTokens =
    modelType match {
      case "roberta" =>
        new SpecialTokens(
          vocab,
          startTokenString = "<s>",
          endTokenString = "</s>",
          unkTokenString = "<unk>",
          maskTokenString = "<mask>",
          padTokenString = "<pad>"
        )
      case "gpt2" =>
        new SpecialTokens(
          vocab,
          startTokenString = "<|endoftext|>",
          endTokenString = "<|endoftext|>",
          unkTokenString = "<|endoftext|>",
          maskTokenString = "<|endoftext|>",
          padTokenString = "<|endoftext|>"
        )
      case "xlm" =>
        new SpecialTokens(
          vocab,
          "<s>",
          "</s>",
          "<unk>",
          "<special1>",
          "<pad>",
          Array("<special0>", "<special2>", "<special3>", "<special4>", "<special5>", "<special6>", "<special7>", "<special8>", "<special9>")
        )
    }
}

case class SpecialToken(content: String, id: Int, singleWord: Boolean = false, lstrip: Boolean = false, rstrip: Boolean = false) {

  override def hashCode(): Int = content.hashCode

  override def canEqual(that: Any): Boolean = that.isInstanceOf[SpecialToken]

  override def equals(obj: Any): Boolean = obj match {
    case obj: SpecialToken => obj.content == content
    case _                 => false
  }

  override def toString: String = content
}
