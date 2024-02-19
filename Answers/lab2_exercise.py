from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Lab 2 Exercise") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

logFile=spark.read.text("./Data/NASA_access_log_Aug95.gz").cache()

### Start mining code



### End mining code

spark.stop()