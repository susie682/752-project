

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CreateSmallCleanedData") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .getOrCreate()

# 读取原始 cleaned_data
df = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data")
df = df.toDF("user_id", "asin", "rating")

# 取前 500 行
x = 3000
small_df = df.limit(x)

# 保存为 HDFS 上新文件
small_df.write.mode("overwrite").csv(f"hdfs://localhost:8020/user/ecommerce_project/cleaned_data_{x}")

spark.stop()