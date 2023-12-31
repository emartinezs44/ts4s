package em.ml.ts4s.tokenizers

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/* Adapted from sparknlp project https://github.com/JohnSnowLabs/spark-nlp */

private abstract class BpeTokenizer(
  merges: Map[(String, String), Int],
  vocab: Map[String, Int],
  specialTokens: SpecialTokens,
  padWithSentenceTokens: Boolean
) {
  protected val bpeRanks: Map[(String, String), Int] = {
    merges
  }

  /** Rankings for the byte pairs. Derived from merges.txt
    */
  protected def getBpeRanking: ((String, String)) => Int =
    (bytePair: (String, String)) => bpeRanks.getOrElse(bytePair, Integer.MAX_VALUE)

  /** cache for already encoded tokens
    */
  protected val cache: mutable.Map[String, Array[String]] = mutable.Map()

  /** Create a sequence of byte-pairs of the word TODO: XLM has to append to end
    */
  protected def getBytePairs(word: Array[String]): Array[(String, String)] = {
    val createPairs = (i: Int) => (word(i), word(i + 1))
    (0 until (word.length - 1)).map(createPairs).toArray
  }

  // Can be overridden in inherited class
  protected val prependForPieceId: Option[String] = None
  protected val appendForPieceId: Option[String]  = None

  protected def performMerges(
    wordChars: Array[String],
    charPairs: Array[(String, String)]
  ): Array[String] = {
    var word  = wordChars
    var pairs = charPairs
    // get highest priority byte-pair first
    var bytePair: (String, String) =
      pairs.sortWith(getBpeRanking(_) < getBpeRanking(_))(0)
    var done = false
    // while we still have byte-pairs from our vocabulary
    while (bpeRanks.contains(bytePair) && !done) {
      val (first, second)             = bytePair
      val newWord: ListBuffer[String] = ListBuffer()
      var i                           = 0
      var j                           = 0
      // keep combining characters with the current byte-pair
      while ((i < word.length) && (j != -1)) {
        j = word.indexOf(first, i)
        if (j == -1) newWord ++= word.drop(i)
        else {
          newWord ++= word.slice(i, j)
          i = j
          val bpIsAtIndex =
            (word(i) == first) && (i < word.length - 1) && word(
              i + 1
            ) == second
          if (bpIsAtIndex) {
            newWord += (first + second)
            i += 2
          } else {
            newWord += word(i)
            i += 1
          }
        }
      }
      word = newWord.toArray
      // if we were able to create a whole word that was in the vocabulary, we're done
      if (word.length == 1) {
        done = true
      } else {
        // do it again with the next byte-pair
        pairs = getBytePairs(word)
        bytePair = pairs.sortWith(getBpeRanking(_) < getBpeRanking(_))(0)
      }
    }
    word
  }

  protected def getTokenPieces(
    indToken: IndexedToken,
    word: Array[String],
    processedToken: String
  ): Array[TokenPiece] = {
    var currentIndex = indToken.begin
    val wordIndexes = word.map((subWord: String) => {
      val startIndex = currentIndex
      currentIndex = startIndex + subWord.length
      (startIndex, startIndex + subWord.length - 1)
    })
    val result = word
      .zip(wordIndexes)
      .map { case (subWord: String, indexes: (Int, Int)) =>
        val isWordStart      = indToken.begin == indexes._1
        var processedSubWord = subWord

        /*
          processedSubWord = prependForPieceId match {
            case None => processedSubWord
            case Some(prepend) =>
              if (isWordStart && subWord.indexOf(prepend) < 0) prepend + processedSubWord
              else processedSubWord
          }*/
        processedSubWord = processedSubWord

        processedSubWord = appendForPieceId match {
          case None => processedSubWord
          case Some(append) =>
            val isWordEnd = indToken.end == indexes._2
            if (isWordEnd && subWord.indexOf(append) < 0)
              processedSubWord + append
            else processedSubWord
        }
        // Set unknown id if not found
        val subWordId: Int =
          vocab.getOrElse(processedSubWord, specialTokens.unk.id)

        TokenPiece(
          subWord,
          processedToken,
          subWordId,
          isWordStart,
          indexes._1,
          indexes._2
        )

      }
    result
  }
  
  protected def bpe(indToken: IndexedToken): Array[TokenPiece] = {
    var processedToken = ""
    try {
      processedToken = preProcessTokenForBpe(indToken.token)
      var word: Array[String] = Array[String]()
      word = processedToken.map(_.toString).toArray
      val pairs: Array[(String, String)] = getBytePairs(word)

      if (pairs.isEmpty)
        word = Array(processedToken)
      else
        word = performMerges(word, pairs)

      getTokenPieces(indToken, word, processedToken)
    } catch {
      case _: java.util.NoSuchElementException =>
        Array(
          TokenPiece(
            indToken.token,
            indToken.token,
            specialTokens.unk.id,
            isWordStart = true,
            indToken.begin,
            indToken.end
          )
        )
    }
  }
  
  protected def splitOnSpecialToken(
    specialToken: SpecialToken,
    text: String
  ): ListBuffer[String] = {
    val isControl = (c: Char) => {
      if (c == '\t' || c == '\n' || c == '\r') false // count as whitespace
      else c.isControl
    }
    val isPunctuation =
      (c: Char) => raw"""[^[:alnum:]]""".r.findFirstIn(c.toString).isDefined
    val isWordBorder =
      (c: Char) => isControl(c) || isPunctuation(c) || c.isWhitespace

    val isEndOfWord   = (text: String) => isWordBorder(text.last)
    val isStartOfWord = (text: String) => isWordBorder(text.head)

    val result: ListBuffer[String] = ListBuffer()
    val tok                        = specialToken.content
    val splitText                  = text.split(tok)
    var fullWord                   = ""
    //    val boolProperty = (property: Map[String, Any], key: String) => property(key).asInstanceOf[Boolean]

    for ((subText, i) <- splitText.zipWithIndex) {
      var done = false
      if (specialToken.singleWord) {
        if (
          (i < (splitText.length - 1)) && !isEndOfWord(
            subText
          ) && !isStartOfWord(splitText(i + 1))
        ) fullWord += subText + tok
        else if (fullWord.nonEmpty) {
          fullWord += subText
          result += fullWord
          fullWord = ""
          done = true
        }
      }
      if (!done) {
        var subTextProcessed: String = subText
        if (specialToken.rstrip && i > 0)
          subTextProcessed = subText.stripPrefix(" ")
        if (specialToken.lstrip && i < (splitText.length - 1))
          subTextProcessed = subText.stripSuffix(" ")
        if (i == 0 && subTextProcessed.isEmpty) result += tok
        else if (i == (splitText.length - 1)) {
          if (subTextProcessed.nonEmpty) result += subTextProcessed
        } else {
          if (subTextProcessed.nonEmpty) result += subTextProcessed
          result += tok
        }
      }
    }
    result
  }
  
  def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken]
  
  val sentencePadding: (String, String) =
    (specialTokens.sentenceStart.content, specialTokens.sentenceEnd.content)
  
  def tokenize(
    sentence: Sentence
  ): Array[IndexedToken] = {
    var text = sentence.content
    if (text.trim.isEmpty) Array[IndexedToken]()
    else {
      val splitTexts: ListBuffer[String] = ListBuffer()
      var textList: ListBuffer[String]   = ListBuffer(text)

      for (transformations <- specialTokens.allTokens) {
        splitTexts.clear()
        for (subText <- textList) {
          if (!specialTokens.contains(subText))
            splitTexts ++= splitOnSpecialToken(transformations, subText)
          else
            splitTexts += subText
        }
        textList = splitTexts.clone()
      }
      if (padWithSentenceTokens) {
        text = sentencePadding._1 + text + sentencePadding._2
        splitTexts.prepend(sentencePadding._1)
        splitTexts.append(sentencePadding._2)
      }
      var currentIndex = 0
      val result       = mutable.ArrayBuffer[IndexedToken]()
      for (subText <- splitTexts) {
        val subTextIndex = sentence.start + text.indexOf(subText, currentIndex)
        if (!specialTokens.contains(subText)) {
          val splitSubText: Array[IndexedToken] =
            tokenizeSubText(subText, subTextIndex)
          result.append(splitSubText: _*)
        } else // subtext is just the special token
           result.append(
             IndexedToken(
               subText,
               begin = subTextIndex,
               end = subTextIndex + subText.length - 1
             )
           )
        currentIndex = subTextIndex + subText.length
      }
      result.toArray
    }
  }

  protected def preProcessTokenForBpe(token: String): String = token

  def encode(indToken: IndexedToken): Array[TokenPiece] = {
    if (!specialTokens.contains(indToken.token))
      bpe(indToken)
    else
      Array(
        TokenPiece(
          indToken.token,
          indToken.token,
          vocab(indToken.token),
          isWordStart = true,
          indToken.begin,
          indToken.end
        )
      )
  }

  def encode(indTokens: Array[IndexedToken]): Array[TokenPiece] =
    indTokens.flatMap(encode(_))
}

object BpeTokenizer {
  def forModel(
    modelType: String,
    merges: Map[(String, String), Int],
    vocab: Map[String, Int],
    padWithSentenceTokens: Boolean = false,
    specialTokens: Option[SpecialTokens] = None
  ): BpeTokenizer = {
    val availableModels = Array("roberta", "xlm")
    require(
      availableModels.contains(modelType),
      "Model type \"" + modelType + "\" not supported yet."
    )

    val modelSpecialTokens = specialTokens match {
      case Some(specialTok) => specialTok
      case None             => SpecialTokens.getSpecialTokensForModel(modelType, vocab)
    }

    modelType match {
      case "roberta" =>
        new RobertaTokenizer(
          merges,
          vocab,
          modelSpecialTokens,
          padWithSentenceTokens
        )
      case _ => throw new Exception("Model not supported!!")
    }
  }
}
