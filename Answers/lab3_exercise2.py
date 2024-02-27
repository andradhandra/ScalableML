from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Lab 3 Exercise 2") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")
print()

# Start code
### Classification With CrossValidator
print("=== Classification With CrossValidator ===")

## Prepare and saperate training and testing data
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]

# Split data to training and testing
(trainingDatag, testDatag) = rawdata.randomSplit([0.7, 0.3], 42)

# Save training and testing data to file
trainingDatag.write.mode("overwrite").parquet('./Data/spamdata_training.parquet')
testDatag.write.mode("overwrite").parquet('./Data/spamdata_test.parquet')
trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')

# Create VectorAssembler to concatenate all features in a vector
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')

    
## Create regularisation instance
lr = LogisticRegression()

# Combine stages into pipeline and create pipeline model
stages = [vecAssembler, lr]
pipeline = Pipeline(stages=stages)

# Create ParamGrid Instance by combining multiple params from diffrent type of Logisitic Regression
paramGrid = ParamGridBuilder() \
            .baseOn({lr.labelCol: 'features'}) \
            .baseOn({lr.labelCol: 'labels'}) \
            .baseOn({lr.maxIter: 50}) \
            .addGrid(lr.regParam, [0, 0.01, 0.01, 0.01]) \
            .addGrid(lr.elasticNetParam, [0, 1, 0, 0.5]) \
            .build()

# Create evaluator instance            
evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

# Create Crossvalidator Instance and embbeding the ParamGridBuilder
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainingData)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(testData)

# Find accuracy
accuracy = evaluator.evaluate(prediction)

## Compare results
# Show accuracy
print("Accuracy = %g " % accuracy)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    print(row)

# End code

spark.stop()