package multilabel

import java.net.URI
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.scheduler.{SparkListener, SparkListenerTaskEnd}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
 * SPRINT.md Sec 2.3 -- plain object, no Estimator/Params/MLWritable conformance.
 * Trains 19 independent binary classifiers (binary relevance) in parallel over a
 * fixed thread pool, sharing one assembled+scaled, cached feature DataFrame.
 *
 * Usage:
 *   spark-submit --class multilabel.BinaryRelevanceRunner ... \
 *     --features-path hdfs://namenode:9000/bigearthnet/features \
 *     --splits-path   hdfs://namenode:9000/bigearthnet/splits/splits.parquet \
 *     --regime s2 --fold "" --learner lr \
 *     --output-path hdfs://namenode:9000/bigearthnet/results --run-id e1_lr_s2
 */
object BinaryRelevanceRunner {

  // Frozen alphabetical class order -- must match src/python/ingest/stream_extract.py CLASSES.
  val CLASSES: Array[String] = Array(
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric"
  )
  require(CLASSES.length == 19)

  val FEATURE_COLS: Array[String] = (0 until 110).map(i => f"f_$i%03d").toArray

  // Never allowed into the VectorAssembler input -- country in particular would leak the answer.
  val METADATA_COLS: Set[String] = Set(
    "patch_id", "tile", "gx", "gy", "month", "year", "country", "split_official",
    "s1_split", "s2_split", "s3_fold", "s4_fold"
  )

  case class Args(
    featuresPath: String = "hdfs://namenode:9000/bigearthnet/features",
    splitsPath: String = "hdfs://namenode:9000/bigearthnet/splits/splits.parquet",
    regime: String = "s2",
    fold: String = "",
    learner: String = "lr",
    outputPath: String = "hdfs://namenode:9000/bigearthnet/results",
    runId: String = "run",
    threadPoolSize: Int = 6,
    seed: Int = 42,
    regParam: Double = 0.01,
    numTrees: Int = 50,
    maxDepth: Int = 10
  )

  def parseArgs(argv: Array[String]): Args = {
    argv.sliding(2, 2).foldLeft(Args()) {
      case (a, Array("--features-path", v))   => a.copy(featuresPath = v)
      case (a, Array("--splits-path", v))     => a.copy(splitsPath = v)
      case (a, Array("--regime", v))          => a.copy(regime = v)
      case (a, Array("--fold", v))            => a.copy(fold = v)
      case (a, Array("--learner", v))         => a.copy(learner = v)
      case (a, Array("--output-path", v))     => a.copy(outputPath = v)
      case (a, Array("--run-id", v))          => a.copy(runId = v)
      case (a, Array("--thread-pool-size", v)) => a.copy(threadPoolSize = v.toInt)
      case (a, Array("--seed", v))             => a.copy(seed = v.toInt)
      case (a, Array("--reg-param", v))        => a.copy(regParam = v.toDouble)
      case (a, Array("--num-trees", v))        => a.copy(numTrees = v.toInt)
      case (a, Array("--max-depth", v))        => a.copy(maxDepth = v.toInt)
      case (a, other) =>
        println(s"WARN: ignoring unrecognized arg pair: ${other.mkString(" ")}")
        a
    }
  }

  def roleColumn(regime: String, fold: String): Column = regime match {
    // fold is normally empty for s1 (uses the seed=42 split); Week 2's split-
    // seed robustness check passes e.g. --fold seed123 to select the
    // s1_split_seed123 column build_splits.py now also writes, reusing this
    // plumbing instead of adding a new CLI flag.
    case "s1" => col(if (fold.isEmpty) "s1_split" else s"s1_split_$fold")
    case "s2" => col("s2_split")
    case "s3" => when(col("s3_fold") === fold.toInt, "test").otherwise("train")
    case "s4" => when(col("s4_fold") === fold, "test").otherwise("train")
    case other => throw new IllegalArgumentException(s"unknown regime: $other (expected s1/s2/s3/s4)")
  }

  def evalRoles(regime: String): Seq[String] =
    if (regime == "s1" || regime == "s2") Seq("validation", "test") else Seq("test")

  def main(argv: Array[String]): Unit = {
    val args = parseArgs(argv)
    require(FEATURE_COLS.forall(c => !METADATA_COLS.contains(c)),
      "metadata column leaked into VectorAssembler input list")

    val spark = SparkSession.builder()
      .appName(s"BR-${args.learner}-${args.regime}-${args.fold}-${args.runId}")
      .config("spark.scheduler.mode", "FAIR")
      .getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    val totalExecutorCpuTimeNs = new AtomicLong(0)
    spark.sparkContext.addSparkListener(new SparkListener {
      override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
        totalExecutorCpuTimeNs.addAndGet(taskEnd.taskMetrics.executorCpuTime)
      }
    })
    val t0 = System.nanoTime()

    val features = spark.read.parquet(args.featuresPath)
    val splits = spark.read.parquet(args.splitsPath)
    val joined = features.join(splits, Seq("patch_id"), "inner")
      .withColumn("role", roleColumn(args.regime, args.fold))
      .persist(StorageLevel.MEMORY_AND_DISK)
    val nJoined = joined.count()
    println(s"Joined features+splits: $nJoined rows")

    // 2. VectorAssembler -> StandardScaler -> persist(MEMORY_AND_DISK), fit ONCE, reused by all 19 labels.
    val assembler = new VectorAssembler().setInputCols(FEATURE_COLS).setOutputCol("features_raw")
    val assembled = assembler.transform(joined)
    val scaler = new StandardScaler()
      .setInputCol("features_raw").setOutputCol("features")
      .setWithMean(true).setWithStd(true)
    // fit scaler on train only, to avoid leaking val/test statistics into feature scaling
    val scalerModel = scaler.fit(assembled.filter(col("role") === "train"))
    val scaled = scalerModel.transform(assembled).persist(StorageLevel.MEMORY_AND_DISK)
    val nTrain = scaled.filter(col("role") === "train").count()
    println(s"Train rows: $nTrain")

    val outRoles = evalRoles(args.regime)
    // Only non-default seeds/hyperparameters get their own path suffix, so
    // every existing (seed=42, default hyperparameter) run's output location
    // is unchanged -- LR is deterministic and never varies by seed; RF seed
    // sweeps and the Week 2 tuning grid need distinct paths.
    val seedSuffix = if (args.seed == 42) "" else s"_seed${args.seed}"
    val hpSuffix =
      if (args.learner == "lr" && args.regParam != 0.01) s"_reg${args.regParam}"
      else if (args.learner == "rf" && (args.numTrees != 50 || args.maxDepth != 10)) s"_t${args.numTrees}d${args.maxDepth}"
      else ""
    val runOutDir = s"${args.outputPath}/predictions/${args.regime}_${args.fold}_${args.learner}$seedSuffix$hpSuffix"

    val pool = Executors.newFixedThreadPool(args.threadPoolSize)
    implicit val ec: ExecutionContext = ExecutionContext.fromExecutorService(pool)

    val futures: Seq[Future[Int]] = (0 until CLASSES.length).map { i =>
      Future {
        val labelCol = f"lbl_$i%02d"
        val trainLabeled = scaled.filter(col("role") === "train")
          .withColumn("label", col(labelCol).cast("double"))

        val counts = trainLabeled.groupBy("label").count().collect()
          .map(r => r.getDouble(0) -> r.getLong(1)).toMap
        val nPos = counts.getOrElse(1.0, 0L).toDouble
        val nNeg = counts.getOrElse(0.0, 0L).toDouble
        val total = nPos + nNeg
        val wPos = if (nPos > 0) total / (2.0 * nPos) else 1.0
        val wNeg = if (nNeg > 0) total / (2.0 * nNeg) else 1.0
        val trainWeighted = trainLabeled.withColumn(
          "weight", when(col("label") === 1.0, wPos).otherwise(wNeg))

        if (nPos == 0.0) {
          println(s"[skip] label=$i (${CLASSES(i)}) has ZERO positives in train " +
            s"for regime=${args.regime} fold=${args.fold} -- cannot fit, skipping")
        } else {
          val model = args.learner match {
            case "lr" =>
              new LogisticRegression()
                .setLabelCol("label").setFeaturesCol("features").setWeightCol("weight")
                .setMaxIter(50).setRegParam(args.regParam).setStandardization(true)
                .fit(trainWeighted)
            case "rf" =>
              new RandomForestClassifier()
                .setLabelCol("label").setFeaturesCol("features").setWeightCol("weight")
                .setNumTrees(args.numTrees).setMaxDepth(args.maxDepth).setSeed(args.seed)
                .fit(trainWeighted)
            case other => throw new IllegalArgumentException(s"unknown learner: $other (expected lr/rf)")
          }

          val probToDouble = udf((v: Vector) => v(1))
          outRoles.foreach { role =>
            val evalDf = scaled.filter(col("role") === role)
              .withColumn("label", col(labelCol).cast("double"))
            val preds = model.transform(evalDf)
            val out = preds.select(
              col("patch_id"),
              lit(role).as("role"),
              lit(i).as("label_idx"),
              lit(CLASSES(i)).as("label_name"),
              col("label").as("true_label"),
              probToDouble(col("probability")).as("probability")
            )
            out.write.mode("overwrite").parquet(f"$runOutDir%s/label_$i%02d/$role%s")
          }
        }
        println(s"[done] label=$i (${CLASSES(i)}) nPos=$nPos nNeg=$nNeg")
        i
      }
    }

    val results = Await.result(Future.sequence(futures), Duration.Inf)
    println(s"Completed ${results.size} labels: ${results.sorted.mkString(",")}")
    pool.shutdown()

    val wallSeconds = (System.nanoTime() - t0) / 1e9
    val computeJson =
      s"""{"run_id":"${args.runId}","regime":"${args.regime}","fold":"${args.fold}",""" +
      s""""learner":"${args.learner}","seed":${args.seed},"reg_param":${args.regParam},""" +
      s""""num_trees":${args.numTrees},"max_depth":${args.maxDepth},"thread_pool_size":${args.threadPoolSize},""" +
      s""""n_joined_rows":$nJoined,"n_train_rows":$nTrain,""" +
      s""""wall_clock_seconds":$wallSeconds,""" +
      s""""total_executor_cpu_time_seconds":${totalExecutorCpuTimeNs.get() / 1e9}}"""

    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val fs = FileSystem.get(new URI(args.outputPath), hadoopConf)
    val outPath = new Path(s"${args.outputPath}/compute/${args.runId}.json")
    val os = fs.create(outPath, true)
    try { os.write(computeJson.getBytes("UTF-8")) } finally { os.close() }
    println(s"Wrote compute stats to ${outPath}")
    println(computeJson)

    spark.stop()
  }
}
