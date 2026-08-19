import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.Pipeline

object LandCoverClassifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("BigEarthNet-LandCover-RF")
      .master("spark://spark-master:7077")
      .getOrCreate()

    val df = spark.read.parquet("hdfs://namenode:9000/bigearthnet/features/features_40k.parquet")

    val labelIndexer = new StringIndexer()
      .setInputCol("dominant_label")
      .setOutputCol("label")

    val featureCols = Array("ndvi", "ndwi", "b02_mean", "b02_std", "b03_mean", "b03_std",
                             "b04_mean", "b04_std", "b08_mean", "b08_std")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // 75/25 split -> 30,000 train / 10,000 test
    val Array(trainDf, testDf) = df.randomSplit(Array(0.75, 0.25), seed = 42)
    println(s"Train: ${trainDf.count()}, Test: ${testDf.count()}")

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(42)
      .setNumTrees(100)
      .setMaxDepth(20)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, rf))

    println("Training heavy Random Forest (100 trees, depth 20)...")
    val model = pipeline.fit(trainDf)
    val predictions = model.transform(testDf)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val f1 = evaluator.evaluate(predictions)
    val accEval = evaluator.setMetricName("accuracy")
    val precEval = evaluator.setMetricName("weightedPrecision")
    val recEval = evaluator.setMetricName("weightedRecall")

    println(s"F1: $f1")
    println(s"Accuracy: ${accEval.evaluate(predictions)}")
    println(s"Precision: ${precEval.evaluate(predictions)}")
    println(s"Recall: ${recEval.evaluate(predictions)}")

    // Export predictions for Power BI / Tableau
    predictions.select("patch_id", "dominant_label", "label", "prediction")
      .write.mode("overwrite").option("header", "true")
      .csv("hdfs://namenode:9000/bigearthnet/results/predictions_csv")

    // Also run baseline Logistic Regression for comparison
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setFamily("multinomial")
    val lrPipeline = new Pipeline().setStages(Array(labelIndexer, assembler, lr))
    val lrModel = lrPipeline.fit(trainDf)
    val lrPreds = lrModel.transform(testDf)
    println(s"Logistic Regression F1: ${evaluator.setMetricName("f1").evaluate(lrPreds)}")
    model.write.overwrite().save("hdfs://namenode:9000/bigearthnet/models/rf_best_model")

    spark.stop()
  }
}
