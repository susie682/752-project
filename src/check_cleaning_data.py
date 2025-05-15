from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("ReadCleanedRecommendationData") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .getOrCreate()

# 读取 HDFS 中的清洗后数据（CSV）
df = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data")

# 添加列名（按保存顺序）
df = df.toDF("user_id", "asin", "rating")

# 展示前几行结果
df.show(1000000, truncate=False)

avg_ratings = df.groupBy("asin").agg(avg("rating").alias("avg_rating"))

# 显示结果（你可以指定具体商品）
avg_ratings.show(100, truncate=False)

avg_ratings = df.groupBy("user_id").agg(avg("rating").alias("avg_rating"))
count = df.groupBy("asin").count().sort('count', ascending=False)
# 显示结果（你可以指定具体商品）
avg_ratings.show(100, truncate=False)
count.show()

# 停止 Spark
spark.stop()