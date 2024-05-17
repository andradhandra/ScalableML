from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# Importing relevant sklearn models
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from pyspark.sql.types import StructField, StructType, DoubleType

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Lab 10 Exercise") \
        .config("spark.local.dir","/fastdata/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

# Enable Arrow-based columnar data transfers.
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Load the dataset and properly set the corresponding dataframe.
rawdata = spark.read.csv('../Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('../Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]
    
# We rename the columns in the dataframe with names of the features in spamd.data.names.
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))
    
# We now create the training and test sets.
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 42)

######## Exercise 1 ########

# We create instances for the vector assembler and the neural network.
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')

mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", maxIter=100, seed=1500)

# Create the pipeline
stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)

evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")

# Create the crossvalidator with the different number of layers and nodes in each layer set in the ParamGrid.
paramGrid = ParamGridBuilder() \
            .addGrid(mpc.layers, [[len(trainingData.columns)-1,20,5,2],  # The first element HAS to be equal to the number of input features.
                                  [len(trainingData.columns)-1,40,10,2],
                                  [len(trainingData.columns)-1,40,20,2],
                                  [len(trainingData.columns)-1,80,20,2],
                                  [len(trainingData.columns)-1,80,40,20,2]]) \
            .build()
        
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

cvModel = crossval.fit(trainingData)

# Predict using the best set of parameters found by the cross validator.
predictions = cvModel.transform(testData)
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)

# .bestModel returns the model object in the crossvalidator. This object is a pipeline.
# .stages[1] returns the second stage in the pipeline, which for our case is our classifier.
# .getParam('layers') returns the Param() object related to the parameter layers that we can use to reference later.
# .extractParamMap() returns a map with the parameters and we reference the parameter layers as if it was a dictionary. 
param = cvModel.bestModel.stages[1].getParam('layers')
bestparams = cvModel.bestModel.stages[1].extractParamMap()[param]
print("Best parameter for layers =", bestparams)

######## Exercise 2 ########

# Convert the Spark DataFrame to a Pandas DataFrame using Arrow
trainingDataPandas = trainingData.select("*").toPandas()

nfeatures = ncolumns-1
Xtrain = trainingDataPandas.iloc[:, 0:nfeatures]
ytrain = trainingDataPandas.iloc[:, -1]

# Different models taken from sklearn (uncomment to use a different model)
#clf = svm.SVC()
#clf = tree.DecisionTreeClassifier(max_depth = 25)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=20, random_state=0)

clf.fit(Xtrain, ytrain)

Xtest = testData.select(spam_names[0:ncolumns-1])
pred_field = [StructField("prediction", DoubleType(), True)] 
new_schema = StructType(Xtest.schema.fields + pred_field)

# We can use the sklearn models directly because they are pickable. Therefore, we do not need the class ModelWrapperPickable from Lab 6
def predict(iterator):
    for features in iterator:
        yield pd.concat([features, pd.Series(clf.predict(features).flatten(), name="prediction")], axis=1)

prediction_sklearn_df = Xtest.mapInPandas(predict, new_schema)
ypred_sklearn = prediction_sklearn_df.select('prediction').toPandas().values

testDataPandas = testData.select("*").toPandas()
ytest = testDataPandas.iloc[:, -1].values

print("Accuracy = %g " % accuracy_score(ypred_sklearn, ytest))

