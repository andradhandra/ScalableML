from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Lab 2 Exercise") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

### Start code

## Log mininng NASA data
# Read log data as text file and processing the columns to match requirement
log_df = spark.read.text("./Data/NASA_access_log_Aug95.gz").cache() \
          .withColumn("fields", F.split(F.col("value"), " ")) \
          .withColumn("host", F.col("fields").getItem(0)) \
          .withColumn("timestamp", F.concat(F.col("fields").getItem(3), F.col("fields").getItem(4))) \
          .withColumn("request", F.concat(F.col("fields").getItem(5), F.col("fields").getItem(6), F.col("fields").getItem(7))) \
          .withColumn("response_code", F.col("fields").getItem(8)) \
          .withColumn("bit_data", F.col("fields").getItem(9))

df = log_df.drop(*["value", "fields"]).cache()

# Print DataFrame
print()
print("Nasa Log File DataFrame")
print("=======================")
df.show(10, False) 
df.printSchema()

# Count unique data in dataframe
unique = df.distinct().count()
print("\n\nHello Spark: There are %i unique data from the log file.\n" % (unique))

# Find the most frequent visitor
grouped = df.groupby("host").count()

print("Number of host visit")
grouped.orderBy(F.desc('count')).show(truncate = False)

frequent_host = grouped.orderBy(F.desc('count')).first()
print("\n\nHello Spark: The most frequent visited host are {} with {} visits.\n".format(frequent_host['host'], frequent_host['count']))


## Regularisation
ad_df = spark.read.load("Data/Advertising.csv", format="csv", inferSchema="true", header="true").drop('_c0').cache()

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

transformed= transData(ad_df)

print("Regularisation")
print("=============")
transformed.show(5)
print()

(trainingData, testData) = transformed.randomSplit([0.6, 0.4], 6012)

# No Regularisation
lr = LinearRegression()
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)

print("No Regularisation Prediction")
predictions.show(5)

# Ordinary least squares Regression
# regParam (λ) = 0, No Penalty
lr1 = LinearRegression(
    maxIter=100, regParam=0.0, elasticNetParam=0.0,
    tol=1e-6, fitIntercept=True, standardization=True, 
    solver="auto", weightCol=None, aggregationDepth=2)
lrModel1 = lr1.fit(trainingData)
predictions1 = lrModel1.transform(testData)

print("Ordinary least squares Regression Prediction")
predictions1.show(5)

# Ridge Regression
# regParam (λ) > 0, elasticNetParam (⍺) = 0,  Penalty L2
lr2 = LinearRegression( 
    maxIter=100, regParam=0.1, elasticNetParam=0.0, 
    tol=1e-6, fitIntercept=True, standardization=True, 
    solver="auto", weightCol=None, aggregationDepth=2)
lrModel2 = lr2.fit(trainingData)
predictions2 = lrModel2.transform(testData)

print("Ridge Regression Prediction")
predictions1.show(5)

# LASSO Regression
# regParam (λ) > 0, elasticNetParam (⍺) = 1,  Penalty L1
lr3 = LinearRegression( 
    maxIter=100, regParam=0.1, elasticNetParam=1.0, 
    tol=1e-6, fitIntercept=True, standardization=True, 
    solver="auto", weightCol=None, aggregationDepth=2)
lrModel3 = lr3.fit(trainingData)
predictions3 = lrModel3.transform(testData)

print("LASSO Regression Prediction")
predictions3.show(5)


# Evaluation
print("Evaluation")
evaluator = RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")

rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data with no Regularisation = %g" % rmse)

rmse1 = evaluator.evaluate(predictions1)
print("Root Mean Squared Error (RMSE) on test data with Ordinary Least Squares Regression = %g" % rmse1)

rmse2 = evaluator.evaluate(predictions2)
print("Root Mean Squared Error (RMSE) on test data with Ridge Regression = %g" % rmse2)

rmse3 = evaluator.evaluate(predictions3)
print("Root Mean Squared Error (RMSE) on test data with LASSO Regression = %g" % rmse3)
print()

## Clssification Pipeline
print("Classification Pipeline")
print("=======================")

training = spark.createDataFrame([
    (0, "a b c d e spark 6012", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h 6012", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

print("Training Data")
training.show()
      
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
logReg = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, logReg])

# Model Fitting
model = pipeline.fit(training)

test = spark.createDataFrame([
    (4, "spyspark hadoop"),
    (5, "spark a b c"),
    (6, "mapreduce spark"),
], ["id", "text"])

print("Test Data")
test.show()

prediction = model.transform(test)
print("Showing Prediction")
prediction.show()

selected = prediction.select("id", "text", "probability", "prediction")
print("show Selected")
selected.show()

print("Result")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))


### End code

spark.stop()