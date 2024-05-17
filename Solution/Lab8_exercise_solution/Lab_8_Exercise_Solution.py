from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib
matplotlib.use('Agg')

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 6 Exercise") \
        .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")


df = spark.read.load("../Data/iris.csv", format="csv", inferSchema="true", header="true").cache()

df.show(20,False)

# extracting true labels
true_labels = df.select('species').cache()

test = [row.species for row in true_labels.collect()]

from pyspark.ml.feature import StringIndexer

string_indexer =  StringIndexer(inputCol = 'species',outputCol = 'label').fit(true_labels)
true_label_indices = string_indexer.transform(true_labels).select('label').collect()
true_label_indices = np.array([row.label for row in true_label_indices])

# training KMeans model witih K=3
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])

dfFeatureVec= transData(df).cache()

myseed = 6012
# set K to 3
kmeans = KMeans().setK(3).setSeed(myseed)

model = kmeans.fit(dfFeatureVec)
predictions = model.transform(dfFeatureVec)

predictions = np.array([row.prediction for row in predictions.select('prediction').collect()])

from sklearn.metrics import normalized_mutual_info_score

score = normalized_mutual_info_score(true_label_indices, predictions)
print(f"when K = 3, seed = {myseed} NMI score is {score}")


seeds = [10,15,20,25,30,35,40,45,50]
results = []
# try different seeds
for seed in seeds:
    kmeans.setSeed(seed)
    model = kmeans.fit(dfFeatureVec)
    predictions = model.transform(dfFeatureVec)

    predictions = np.array([row.prediction for row in predictions.select('prediction').collect()])

    score = normalized_mutual_info_score(true_label_indices, predictions)
    print(f"with seed {seed} NMI score is {score}")
    results.append(score)

# plot results
fig, ax = plt.subplots()
rects = ax.bar([str(r) for r in seeds], results, label = "NMI score")

ax.set_ylabel('NMI score result')
ax.set_title("NMI score by seed")
ax.yaxis.set_data_interval(min(results), max(results),True)
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.savefig("../Output/Lab6_plot.png")