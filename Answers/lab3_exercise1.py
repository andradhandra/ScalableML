from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Lab 3 Exercise 1") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Start code

### Classification Regularisation
print("=== Classification Regularisation ===")

## Prepare and saperate training and testing data
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

# For being able to save files in a Parquet file format, later on, we need to rename
# two columns with invalid characters ; and (
spam_names[spam_names.index('char_freq_;')] = 'char_freq_semicolon'
spam_names[spam_names.index('char_freq_(')] = 'char_freq_leftp'

# Renaming columns to 'labels'
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

# Change column data type to Double
for i in range(ncolumns):
    rawdata = rawdata.withColumn(spam_names[i], rawdata[spam_names[i]].cast(DoubleType()))

# Split data to training and testing
(trainingDatag, testDatag) = rawdata.randomSplit([0.7, 0.3], 42)

# Save training and testing data to file
trainingDatag.write.mode("overwrite").parquet('./Data/spamdata_training.parquet')
testDatag.write.mode("overwrite").parquet('./Data/spamdata_test.parquet')
trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')

# Create VectorAssembler to concatenate all features in a vector
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')



## Create L2 regularisation
lrL2 = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, elasticNetParam=0, family="binomial")

# Combine stages into pipeline and create pipeline model
stageslrL2 = [vecAssembler, lrL2]
pipelinelrL2 = Pipeline(stages=stageslrL2)
pipelineModellrL2 = pipelinelrL2.fit(trainingData)

# Compute the accuracy
predictionslrL2 = pipelineModellrL2.transform(testData)
evaluatorlrL2 = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracylrL2 = evaluatorlrL2.evaluate(predictionslrL2)

# Save vector w obtained without regularisation
w_L2 = pipelineModellrL2.stages[-1].coefficients.values

# Find preferred features
L2pref = spam_names[np.argmax(np.abs(w_L2))]

# Summary method to find accuracy
lrModelL2 = pipelineModellrL2.stages[-1]
sumAccL2 = lrModelL2.summary.accuracy



## Create elastic net regularisation
lrEN = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, elasticNetParam=0.5, family="binomial")

# Combine stages into pipeline and create pipeline model
stageslrEN = [vecAssembler, lrEN]
pipelinelrEN = Pipeline(stages=stageslrEN)
pipelineModellrEN = pipelinelrEN.fit(trainingData)

# Compute the accuracy
predictionslrEN = pipelineModellrEN.transform(testData)
evaluatorlrEN = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracylrEN = evaluatorlrEN.evaluate(predictionslrEN)

# Save vector w obtained without regularisation
w_EN = pipelineModellrEN.stages[-1].coefficients.values

# Find preferred features
ENpref = spam_names[np.argmax(np.abs(w_EN))]

# Summary method to find accuracy
lrModeEN = pipelineModellrEN.stages[-1]
sumAccEN = lrModeEN.summary.accuracy

## Compare results
# Compare accuracy
print("Logisitic Regression with L2 accuracy = %g " % accuracylrL2)
print("Logisitic Regression with Elastic Net accuracy = %g " % accuracylrEN)

# Compare preferred features
print("L2 preffered features: %a" % L2pref)
print("Elastic Net preffered features: %a" % ENpref)

# Compare summary accuracy
print("LRL2 summary accuracy = %g " % sumAccL2)
print("LREN summary accuracy = %g " % sumAccEN)

# End code

spark.stop()