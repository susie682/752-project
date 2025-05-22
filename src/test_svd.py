from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, row_number
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import ALS as SVD_ALS, Rating
import time

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("RunSVDFromMappedData") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

sc = spark.sparkContext
x = 1000000
# 开始计时


# 读取用户和商品映射表
user_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_user_id") \
    .toDF("user_id", "userIndex")
item_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_asin_id") \
    .toDF("asin", "itemIndex")

# 读取评分数据（使用小数据集）
ratings = spark.read.option("header", "false").csv(f"hdfs://localhost:8020/user/ecommerce_project/cleaned_data_{x}") \
    .toDF("user_id", "asin", "rating")
ratings = ratings.withColumn("rating", col("rating").cast("float"))

# 连接映射得到整数 ID
indexed = ratings.join(user_ids, on="user_id", how="inner") \
                 .join(item_ids, on="asin", how="inner") \
                 .select(
                     col("userIndex").cast("int").alias("userId"),
                     col("itemIndex").cast("int").alias("productId"),
                     col("rating")
                 )

indexed.cache()



ratings_rdd = indexed.rdd.map(lambda row: Rating(row["userId"], row["productId"], row["rating"]))

start_time = time.time()
training_rdd, test_rdd = ratings_rdd.randomSplit([0.9, 0.1], seed=42)


model = SVD_ALS.train(training_rdd, rank=30, iterations=20, lambda_=0.1)

# 回退机制：若测试集为空则使用训练集
if test_rdd.isEmpty():
    print("❗ No test data. Using training set instead.")
    test_user_product = training_rdd.map(lambda r: (r[0], r[1]))
    rates = training_rdd.map(lambda r: ((r[0], r[1]), r[2]))
else:
    test_user_product = test_rdd.map(lambda r: (r[0], r[1]))
    rates = test_rdd.map(lambda r: ((r[0], r[1]), r[2]))

# 预测并计算 RMSE
predictions = model.predictAll(test_user_product).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = rates.join(predictions)
mse = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
rmse = mse ** 0.5
print(f"data count = {x}")
print(f"📉SVD  RMSE = {rmse:.3f}")



# 计时结束
end_time = time.time()
print(f"⏱️ SVD recommendation time cost：{end_time - start_time:.2f} Seconds")



# 停止 Spark
spark.stop()