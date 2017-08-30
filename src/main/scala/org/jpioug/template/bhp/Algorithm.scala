package org.jpioug.template.bhp

import org.apache.predictionio.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.jpioug.template.python.{Engine, PreparedData}

case class AlgorithmParams(name: String) extends Params

case class Query(AGE: Double,
                 B: Double,
                 CHAS: Double,
                 CRIM: Double,
                 DIS: Double,
                 INDUS: Double,
                 LSTAT: Double,
                 NOX: Double,
                 PTRATIO: Double,
                 RAD: Double,
                 RM: Double,
                 TAX: Double,
                 ZN: Double)

case class PredictedResult(label: Double) extends Serializable

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, PipelineModel, Query, PredictedResult] {

  def train(sc: SparkContext, data: PreparedData): PipelineModel = {
    Engine.modelRef.get()
  }

  def predict(model: PipelineModel, query: Query): PredictedResult = {
    val spark = SparkSession
      .builder
      .appName(ap.name)
      .getOrCreate()
    import spark.implicits._
    val data = Seq((
      query.AGE,
      query.B,
      query.CHAS,
      query.CRIM,
      query.DIS,
      query.INDUS,
      query.LSTAT,
      query.NOX,
      query.PTRATIO,
      query.RAD,
      query.RM,
      query.TAX,
      query.ZN
    ))
    val df = spark.createDataset(data).toDF("AGE",
      "B",
      "CHAS",
      "CRIM",
      "DIS",
      "INDUS",
      "LSTAT",
      "NOX",
      "PTRATIO",
      "RAD",
      "RM",
      "TAX",
      "ZN")
    val labelDf = model.transform(df)
    PredictedResult(labelDf.select("prediction").first().getAs(0))
  }
}

