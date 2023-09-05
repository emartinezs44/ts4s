package em.ml.ts4s
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object TestingSpark {

  given spark: SparkSession = SparkSession.builder().master("local[1]").getOrCreate()
  import spark.implicits.*

  def main(args: Array[String]): Unit = {
    val seq = Seq("one", "two")
    spark.createDataset[String](seq).show()
  }
}
