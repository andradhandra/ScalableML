from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoder

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Lab 3 Exercise 2") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")
print()

# Start code
### Scalable GLM
print("=== Scalable GLM ===")

## Prepare data frame
rawdata = spark.read.csv('./Data/hour.csv', header=True)
rawdata.cache()

schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
new_rawdata = rawdata.select(schemaNames[2:ncolumns])

new_schemaNames = new_rawdata.schema.names
new_ncolumns = len(new_rawdata.columns)
for i in range(new_ncolumns):
    new_rawdata = new_rawdata.withColumn(new_schemaNames[i], new_rawdata[new_schemaNames[i]].cast(DoubleType()))

## Transform categorical value into one-hot encoding
categorical_schemaNames = new_schemaNames[:8]
encoded_schemaNames = ['encoded_'+col for col in categorical_schemaNames]
ohe = OneHotEncoder(inputCols = categorical_schemaNames, outputCols = encoded_schemaNames)
encoded_model = ohe.fit(new_rawdata)
encoded_rawdata = encoded_model.transform(new_rawdata)
# encoded_rawdata.show(truncate=False)

# Split encoded data into training and test
(trainingData, testData) = encoded_rawdata.randomSplit([0.7, 0.3], 42)

# Assemble the features into a vector
assembler = VectorAssembler(inputCols = encoded_schemaNames+new_schemaNames[8:12], outputCol = 'features') 

# Create Poisson Regression Instance
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01, family='poisson', link='log')

# Create Linear Regression with l1 regularisation
lr1_auto = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=1.0, solver="auto")
lr1_normal = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=1.0, solver="normal")

# Create Linear Regression with l2 regularisation
lr2_auto = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=0.0, solver="auto")
lr2_normal = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=0.0, solver="normal")
lr2_lbfgs = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=0.0, solver="l-bfgs")

# Create Linear Regression with elastic net regularisation
en_auto = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=0.5, solver="auto")
en_normal = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.1, elasticNetParam=0.5, solver="normal")


# Create poisson pipeline and calculate RMSE
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of GLM poisson model ===> %g " % rmse)

# Create lr1 auto pipeline and calculate RMSE
stages = [assembler, lr1_auto]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of lr1 auto ===> %g " % rmse)

# Create lr1 normal and alculate RMSE
stages = [assembler, lr1_normal]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of lr1 normal ===> %g " % rmse)

# Create lr2 auto and alculate RMSE
stages = [assembler, lr2_auto]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of lr2 auto ===> %g " % rmse)

# Create lr2 normal and alculate RMSE
stages = [assembler, lr2_normal]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print("RMSE of lr2 normal ===> %g " % rmse)

# Create lr2 l-bfgs and alculate RMSE
stages = [assembler, lr2_lbfgs]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of lr2 l-bfgs ===> %g " % rmse)

# Create elastic net auto and alculate RMSE
stages = [assembler, en_auto]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of elastic net auto ===> %g " % rmse)

# Create elastic net normal and alculate RMSE
stages = [assembler, en_normal]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE of elastic net normal ===> %g " % rmse)


# End code

spark.stop()