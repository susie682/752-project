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

# 开始计时
start_time = time.time()

# 读取用户和商品映射表
user_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_user_id") \
    .toDF("user_id", "userIndex")
item_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_asin_id") \
    .toDF("asin", "itemIndex")

# 读取评分数据（使用小数据集）
ratings = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/cleaned_data_small") \
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
indexed.printSchema()
indexed.show()


ratings_rdd = indexed.rdd.map(lambda row: Rating(row["userId"], row["productId"], row["rating"]))


training_rdd, test_rdd = ratings_rdd.randomSplit([0.8, 0.2], seed=42)


model = SVD_ALS.train(training_rdd, rank=10, iterations=10, lambda_=0.1)

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
print(f"📉SVD  RMSE = {rmse:.3f}")

# 推荐每位用户 3 个商品
userRecs = model.recommendProductsForUsers(3)

# 计时结束
end_time = time.time()
print(f"⏱️ SVD recommendation time cost：{end_time - start_time:.2f} Seconds")

# 扁平化推荐结果为 DataFrame
flatRecs = userRecs.flatMapValues(lambda recs: recs) \
    .map(lambda x: (x[0], x[1].product, x[1].rating)) \
    .toDF(["userId", "productId", "predictedRating"])

# 去除用户已评分商品
user_item_df = indexed.select("userId", "productId")
final_recommendations = flatRecs.join(user_item_df, on=["userId", "productId"], how="left_anti")

# 每位用户保留 top 3
windowSpec = Window.partitionBy("userId").orderBy(col("predictedRating").desc())
ranked = final_recommendations.withColumn("rank", row_number().over(windowSpec))
topN = ranked.filter(col("rank") <= 3)

# 展示推荐结果
topN.select("userId", "productId", "predictedRating").show(truncate=False)

# 停止 Spark
spark.stop()