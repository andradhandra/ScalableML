from pyspark.sql import SparkSession
import pyspark.sql.functions as F

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
print("====================")
grouped.orderBy(F.desc('count')).show(truncate = False)

frequent_host = grouped.orderBy(F.desc('count')).first()
print("\n\nHello Spark: The most frequent visited host are {} with {} visits.\n".format(frequent_host['host'], frequent_host['count']))

### End code

spark.stop()