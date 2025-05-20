from pyspark.sql import SparkSession

# 初始化 SparkSession（本地模式 + 设置 HDFS 地址）
spark = SparkSession.builder \
    .appName("CleanRecommendationData") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .getOrCreate()

# 从 HDFS 读取 JSONL 原始数据
df = spark.read.json("hdfs://localhost:8020/user/ecommerce_project/data/Handmade_Products.jsonl")

# 提取需要字段并去除空值
clean_df = df.select("user_id", "asin", "rating").dropna()

# 显示前几行用于验证
clean_df.show(10, truncate=False)

# 保存为 CSV 到 HDFS
clean_df.write.mode("overwrite").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data")

# 停止 Spark 会话
spark.stop()

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CreateSmallCleanedData") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .getOrCreate()

# 读取原始 cleaned_data
df = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data")
df = df.toDF("user_id", "asin", "rating")

# 取前 500 行
small_df = df.limit(500)

# 保存为 HDFS 上新文件
small_df.write.mode("overwrite").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data_small")

spark.stop()