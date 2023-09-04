package em.ml.ts4s.dllib.nlp.models

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, MultiLabelSoftMarginCriterion}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity, TensorCriterion}
import com.intel.analytics.bigdl.dllib.optim.{OptimMethod, ParallelAdam, Top1Accuracy, ValidationMethod}
import com.intel.analytics.bigdl.dllib.tensor.{DenseTensor, Tensor}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import em.ml.ts4s.dllib.util.LoadWeightsAndBiases
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

trait DlModel {

  def unfreeze: List[String]

  def fit(
    checkpointPath: String = "",
    trainingDataset: RDD[Sample[Float]],
    validationDataset: RDD[Sample[Float]],
    batchSize: Int,
    epochs: Int,
    outputModelPath: String,
    outputWeightsPath: String,
    categoryColumn: String,
    tokensColumn: String,
    modelPath: String = "",
    weightsPath: String = "",
    lr: Double = 5e-5,
    wDecay: Double = 0.0,
    lrDecay: Double = 0.0,
    inputModel: Option[AbstractModule[Activity, Activity, Float]] = None,
    multiLabel: Boolean = false,
    validationPerEpoch: Boolean = true,
    parallelAdam: Boolean = false,
    isLora: Boolean = false
  )(using ev: TensorNumeric[Float]): KerasNet[Float] = {

    val model = {
      if (inputModel.isDefined) inputModel.get.asInstanceOf[KerasNet[Float]]
      else {
        require(
          modelPath.nonEmpty && weightsPath.nonEmpty,
          "Invalid parameters. You must include model and weights paths"
        )
        LoadWeightsAndBiases.loadModel(modelPath, weightsPath)
      }
    }

    val criterion = new ClassNLLCriterion[Float]
    val optimizer = new Adam[Float](
      lr = lr,
      decay = lrDecay,
      wDecay = wDecay,
      epsilon = 1e-6
    )

    if (!checkpointPath.isEmpty)
      model.setCheckpoint(checkpointPath)

    model.compileCompat(
      optimizer,
      criterion,
      List[ValidationMethod[Float]]()
    )

    if (isLora)
       model.freeze()
       model.unFreeze(unfreeze: _*)


    model.summary()
    model.fit(trainingDataset, batchSize, epochs, validationDataset)
    model
  }
}
