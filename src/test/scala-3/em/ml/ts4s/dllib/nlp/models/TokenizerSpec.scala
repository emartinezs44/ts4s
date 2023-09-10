package em.ml.ts4s.dllib.nlp.models

import org.junit.{Assert, Test}

class TokenizerSpec {
  @Test
  def checkTokensSpec(): Unit = {
    val result = RobertaForSequenceClassification.tokenizeStr("_Buenos días.!!")
    Assert.assertArrayEquals(result, Array(117, 19791, 1243, 68, 1240))
  }
}
