package em.ml.ts4s.tokenizers

import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

/* Adapted from sparknlp project https://github.com/JohnSnowLabs/spark-nlp */
class RobertaTokenizer(
  merges: Map[(String, String), Int],
  vocab: Map[String, Int],
  specialTokens: SpecialTokens,
  padWithSentenceTokens: Boolean = false
) extends BpeTokenizer(
    merges,
    vocab,
    specialTokens,
    padWithSentenceTokens
  ) {

  private val bytesToUnicodeMapping: Map[Int, String] = {
    val bytes: ListBuffer[Int] = ListBuffer.range[Int](
      '!',
      '~' + 1
    ) ++ ListBuffer.range[Int]('¡', '¬' + 1) ++ ListBuffer.range[Int]('®', 'ÿ' + 1)
    val characters: ListBuffer[Int] = bytes.clone
    var n                           = 0
    for (b <- 0 to 256) {
      if (!bytes.contains(b)) {
        bytes += b
        characters += (256 + n)
        n += 1
      }
    }
    (bytes zip characters.map(_.toChar.toString)).toMap
  }

  override val prependForPieceId: Option[String] = Some("Ġ")

  override def preProcessTokenForBpe(token: String): String =
    token.foldLeft("")(_ + bytesToUnicodeMapping(_))

  val splitPattern: Regex =
    raw"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".r

  override def tokenizeSubText(
    text: String,
    indexOffset: Int
  ): Array[IndexedToken] = {
    // split pattern based on gpt2's bpe tokenizer
    splitPattern
      .findAllMatchIn(text)
      .map(tok =>
        IndexedToken(
          tok.matched,
          tok.start + indexOffset,
          tok.end + indexOffset - 1
        )
      )
      .toArray
  }
}
