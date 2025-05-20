from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, row_number
from pyspark.sql.window import Window
import time

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("RunALSFromMappedData") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

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

# 缓存并输出调试信息
ratings.cache()
ratings.printSchema()
ratings.show()

# 加入索引，构造 ALS 输入格式
indexed = ratings.join(user_ids, on="user_id", how="inner") \
                 .join(item_ids, on="asin", how="inner") \
                 .select(col("userIndex").cast("int"), col("itemIndex").cast("int"), col("rating"))

# 拆分训练 / 测试集
training = indexed.sample(False, 0.8, seed=42)
test = indexed.subtract(training)

# 构建并训练 ALS 模型
als = ALS(
    maxIter=10,
    regParam=0.1,
    rank=10,
    userCol="userIndex",
    itemCol="itemIndex",
    ratingCol="rating",
    coldStartStrategy="drop"
)
model = als.fit(training)

# 预测评分：若 test 无结果则回退 training
predictions = model.transform(test)
if predictions.rdd.isEmpty():
    print("❗ No predictions generated. Using training set instead.")
    predictions = model.transform(training)

predictions.show()

# 评估 RMSE
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"📉 RMSE = {rmse:.3f}")

# 每位用户推荐 3 个商品
user_recs = model.recommendForAllUsers(3)
print("🔮 Top 3 Recommendations for Each User:")
user_recs.show(truncate=False)

# 推荐流程计时结束
end_time = time.time()
print(f"⏱️ ALS 推荐流程耗时：{end_time - start_time:.2f} 秒")

# 扁平化推荐结果
exploded = user_recs.select("userIndex", explode("recommendations").alias("rec"))
recommendations = exploded.select(
    col("userIndex"),
    col("rec.itemIndex").alias("itemIndex"),
    col("rec.rating").alias("predictedRating")
)

# 去除已评分商品
user_item_df = indexed.select("userIndex", "itemIndex")
final_recommendations = recommendations.join(user_item_df, on=["userIndex", "itemIndex"], how="left_anti")

# 每位用户保留 top 3 推荐
windowSpec = Window.partitionBy("userIndex").orderBy(col("predictedRating").desc())
ranked = final_recommendations.withColumn("rank", row_number().over(windowSpec))
topN = ranked.filter(col("rank") <= 3)

# 展示推荐结果
topN.select("userIndex", "itemIndex", "predictedRating").show(truncate=False)

# 停止 Spark
spark.stop()