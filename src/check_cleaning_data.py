from pyspark import Row
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
df.show(100, truncate=False)
# 假设你已经有 df，包含 user_id, asin, rating（字符串）

# 生成 user 映射表
user_ids = df.select("user_id").distinct().rdd \
    .map(lambda r: r[0]) \
    .zipWithIndex() \
    .map(lambda x: Row(user_id=x[0], userIndex=x[1])) \
    .toDF()

# 生成 asin 映射表（即 product id）
item_ids = df.select("asin").distinct().rdd \
    .map(lambda r: r[0]) \
    .zipWithIndex() \
    .map(lambda x: Row(asin=x[0], itemIndex=x[1])) \
    .toDF()

item_ids.write.mode("overwrite").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_asin_id")
avg_ratings = df.groupBy("asin").agg(avg("rating").alias("avg_rating"))

# 显示结果（你可以指定具体商品）
# 读取原始数据
df = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data")
df = df.toDF("user_id", "asin", "rating")

# 读取映射表
user_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_user_id") \
    .toDF("user_id", "userIndex")
item_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_asin_id") \
    .toDF("asin", "itemIndex")

# join 并转为数值列
ratings = df.join(user_ids, on="user_id", how="left") \
            .join(item_ids, on="asin", how="left") \
            .selectExpr("cast(userIndex as int) as userIndex",
                        "cast(itemIndex as int) as itemIndex",
                        "cast(rating as float) as rating")