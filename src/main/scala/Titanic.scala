import org.apache.log4j.{Logger, Level}
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, Imputer, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object Titanic {

  val trainFile = "data/train.csv"
  val testFile = "data/test.csv"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Titanic")
      .getOrCreate

    import spark.implicits._

    //Loading data
    var trainDF = loadDF(trainFile, spark)
    var testDF = loadDF(testFile, spark)

    //Adding isAlone feature
    val isAlone: ((Int, Int) => Double) = (sibSp: Int, parCh: Int) => if (sibSp + parCh > 0) 0.0 else 1.0
    val isAloneUDF = udf(isAlone)

    trainDF = trainDF
      .withColumn("IsAlone", isAloneUDF(col("SibSp"), col("Parch")))
    testDF = testDF
      .withColumn("IsAlone", isAloneUDF(col("SibSp"), col("Parch")))

    //Filling na Embarked with "S"
    val fillEmbarked = Map("Embarked" -> "S")

    trainDF = trainDF
      .na.fill(fillEmbarked)
    testDF = testDF
      .na.fill(fillEmbarked)

    //Transforming Survived to double
    val toDouble: (Int => Double) = (n: Int) => n.toDouble
    val toDoubleUDF = udf(toDouble)

    trainDF = trainDF
      .withColumn("Survived", toDoubleUDF(col("Survived")))

    //Building pipeline
    val pclassIndexer = new StringIndexer()
      .setInputCol("Pclass")
      .setOutputCol("PclassIndexed")

    val sexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndexed")

    val embarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndexed")

    val imputer = new Imputer()
      .setInputCols(Array("Age", "Fare"))
      .setOutputCols(Array("AgeImputed", "FareImputed"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("PclassIndexed", "SexIndexed", "EmbarkedIndexed", "AgeImputed", "FareImputed", "IsAlone"))
      .setOutputCol("Features")

    val randomForest = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("Features")
      .setNumTrees(10)

    val pipeline = new Pipeline()
      .setStages(Array(pclassIndexer, sexIndexer, embarkedIndexer, imputer, assembler, randomForest))

    //Cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth, Array(4, 6, 8))
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    //Evaluating model
    val splits = trainDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()

    val model = cv.fit(train)
    var result = model.transform(test)
    result = result.select("prediction", "Survived")
    val predictionAndLabels = result.map {
      row => (row.get(0).asInstanceOf[Double], row.get(1).asInstanceOf[Double])
    }.rdd

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println("Area under ROC = " + metrics.areaUnderROC())

    //Making a prediction
    val cvModel = cv.fit(trainDF)
    var prediction = cvModel.transform(testDF)

    //Determining which class has highest survival rate among men
    val maleUpperClass = prediction.filter("Pclass == 1").filter("Sex == 'male'")
    val maleUpperClassSurvived = maleUpperClass.filter("prediction == 1.0")
    val percentageUpper = maleUpperClassSurvived.count().toDouble/maleUpperClass.count().toDouble

    val maleMiddleClass = prediction.filter("Pclass == 2").filter("Sex == 'male'")
    val maleMiddleClassSurvived = maleMiddleClass.filter("prediction == 1.0")
    val percentageMiddle = maleMiddleClassSurvived.count().toDouble/maleMiddleClass.count().toDouble

    val maleLowerClass = prediction.filter("Pclass == 3").filter("Sex == 'male'")
    val maleLowerClassSurvived = maleLowerClass.filter("prediction == 1.0")
    val percentageLower = maleLowerClassSurvived.count().toDouble/maleLowerClass.count().toDouble

    case class Pair(name: String, value: Double)
    val maleSurvivalRateByClasses = Array(
      Pair("Upper", percentageUpper),
      Pair("Middle", percentageMiddle),
      Pair("Lower", percentageLower)
    )

    println("Upper class male - total: " + maleUpperClass.count()
      + " - survived: "+ maleUpperClassSurvived.count() + " - percentage: " + percentageUpper)

    println("Middle class male - total: " + maleMiddleClass.count()
      + " - survived: "+ maleMiddleClassSurvived.count() + " - percentage: " + percentageMiddle)

    println("Lower class male - total: " + maleLowerClass.count()
      + " - survived: "+ maleLowerClassSurvived.count() + " - percentage: " + percentageLower)

    println(maleSurvivalRateByClasses.maxBy(_.value).name + " class has the highest survival rate among men.")
  }

  def loadDF(fileName: String, spark: SparkSession): DataFrame = {
    val df = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(fileName)

    df
  }

}