package em.ml.ts4s.dllib.nlp.models
/*
import com.intel.analytics.bigdl.dllib.utils.{MultiShape, Shape}
import org.scalatest.*
import flatspec.*
import matchers.*

import scala.reflect.ClassTag

class RobertaBaseForwardSpec extends AnyFlatSpec with should.Matchers {

  val seqLen            = 514
  val bertBaseSeqLength = 514
  val hiddenSize        = 768

  val bertBaseShape = MultiShape(
    List(
      Shape(1, bertBaseSeqLength),
      Shape(1, bertBaseSeqLength),
      Shape(1, 1, 1, bertBaseSeqLength)
    )
  )

  "Roberta layer" should "respond in a specific time" in {

    /*val input = ExampleData.getSampleTensor1(512)
    val layer = new RobertaBase[Float](
      vocab = 50262,
      hiddenSize = hiddenSize,
      nBlock = 12,
      nHead = 12,
      intermediateSize = 3072,
      hiddenPDrop = 0.1,
      attnPDrop = 0.1,
      maxPositionLen = bertBaseSeqLength,
      outputAllBlock = false,
      inputSeqLen = bertBaseSeqLength,
      headLayer = None
    )

    val model = layer.doBuild(bertBaseShape).evaluate()
    val t      = System.nanoTime
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    model.forward(input)
    //model.backward(input, result)
    val duration = (System.nanoTime - t) / 1e9d
    println("duration: " + duration)*/

    val input = ExampleData.getSampleTensor1(514)
    val model1 = new RobertaBase[Float](
      vocab = 50262,
      hiddenSize = hiddenSize,
      nBlock = 12,
      nHead = 12,
      intermediateSize = 3072,
      hiddenPDrop = 0.1,
      attnPDrop = 0.1,
      maxPositionLen = bertBaseSeqLength,
      outputAllBlock = false,
      inputSeqLen = bertBaseSeqLength,
      headLayer = None
    )
    val model2 = model1.evaluate()
    model2.build(bertBaseShape)


    val result = model2.evaluate()
    val t = System.nanoTime
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    result.forward(input)
    val duration = (System.nanoTime - t) / 1e9d
    println("D: " + duration)

  }

  "Roberta layer" should "respond in a specific time0" in {

    val seqLen = 514
    val bertBaseSeqLength = 514
    val hiddenSize = 768

    /** At the moment NO Seq Len */
    val bertBaseShape = Shape(
      List(
        Shape(1, bertBaseSeqLength),
        Shape(1, bertBaseSeqLength),
        Shape(1, 1, 1, bertBaseSeqLength)
      )
    )

    val input = getSampleTensor1(514)

    val model1 = new RobertaBase[Float](
      vocab = 50262,
      hiddenSize = hiddenSize,
      nBlock = 12,
      nHead = 12,
      intermediateSize = 3072,
      hiddenPDrop = 0.1,
      attnPDrop = 0.1,
      maxPositionLen = bertBaseSeqLength,
      outputAllBlock = false,
      inputSeqLen = bertBaseSeqLength,
      headLayer = None
    )

    val model2 = model1.evaluate()
    model2.build(bertBaseShape)
    val result = model2.evaluate()

    val t = System.nanoTime
    result.forward(input)
    val duration = (System.nanoTime - t) / 1e9d
    println("D: " + duration)
    val t0 = System.nanoTime
    result.forward(input)
    val duration1 = (System.nanoTime - t0) / 1e9d
    println("D: " + duration1)

    val t1 = System.nanoTime
    result.forward(input)
    val duration2 = (System.nanoTime - t1) / 1e9d
    println("D: " + duration2)

    val t2 = System.nanoTime
    result.forward(input)
    val duration3 = (System.nanoTime - t2) / 1e9d
    println("D: " + duration3)

    val t3 = System.nanoTime
    result.forward(input)
    val duration4 = (System.nanoTime - t3) / 1e9d
    println("D: " + duration4)
  }


}*/
