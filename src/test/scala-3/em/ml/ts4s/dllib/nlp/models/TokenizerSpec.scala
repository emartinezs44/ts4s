package em.ml.ts4s.dllib.nlp.models

import org.junit.Test

class TokenizerSpec {
  @Test
  def tokenizerSpec(): Unit = {

    val result = RobertaForSequenceClassification.tokenizeStr("_Buenos días.!!")
    println(result.toList)

  }
}
