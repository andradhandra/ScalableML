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
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Lab 3 Exercise 1") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")
print()

# Start code
### Classification With CrossValidator
print("=== Classification With CrossValidator ===")

# Use saved training and testing data from file
trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')



## Init Logisitic Regression
lr = LogisticRegression(featuresCol='features', labelCol='labels')
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(trainingData)

# Combine stages into pipeline and create pipeline model
stageslrL2 = [vecAssembler, lrL2]
pipelinelrL2 = Pipeline(stages=stageslrL2)
pipelineModellrL2 = pipelinelrL2.fit(trainingData)

# Compute the accuracy
predictionslrL2 = pipelineModellrL2.transform(testData)
accuracylrL2 = evaluatorlrL2.evaluate(predictionslrL2)

# Save vector w obtained without regularisation
w_L2 = pipelineModellrL2.stages[-1].coefficients.values

# Find preferred features
L2pref = spam_names[np.argmax(np.abs(w_L2))]

# Summary method to find accuracy
lrModelL2 = pipelineModellrL2.stages[-1]
sumAccL2 = lrModelL2.summary.accuracy

## Compare results
# Compare accuracy
print("Logisitic Regression with L2 accuracy = %g " % accuracylrL2)

# Compare preferred features
print("L2 preffered features: %a" % L2pref)

# Compare summary accuracy
print("LRL2 summary accuracy = %g " % sumAccL2)

# End code

spark.stop()