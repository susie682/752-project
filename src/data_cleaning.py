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