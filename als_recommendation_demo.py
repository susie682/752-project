from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode
import time  

# ✅ 1. 创建 SparkSession，并连接 HDFS（注意 fs.defaultFS 配置）
spark = SparkSession.builder \
    .appName("ALS_ECommerce_Recommendation") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ✅ 2. 从 HDFS 加载 CSV 格式的评分数据（userId, productId, rating）
ratings = spark.read.csv("hdfs://localhost:9000/user/suzy/ratings_large.csv", header=True, inferSchema=True)

# ✅ 加这两行：调试用，打印字段结构和内容
ratings.printSchema()
ratings.show()

# ✅ 3. 缓存数据，方便后续多次使用
ratings.cache()
ratings.show()


# ✅ 在拆分训练集/测试集前开始计时
start_time = time.time()

# ✅ 4. 拆分训练集 / 测试集，避免 cold start 问题
training = ratings.sample(False, 0.8, seed=42)
test = ratings.subtract(training)

# ✅ 5. 构建 ALS 模型
als = ALS(
    userCol="userId",
    itemCol="productId",
    ratingCol="rating",
    maxIter=10,
    regParam=0.1,
    rank=10,
    coldStartStrategy="drop"  # 避免预测值为 NaN
)

# ✅ 6. 拟合模型
model = als.fit(training)

# ✅ 7. 在测试集上预测评分
predictions = model.transform(test)

# ✅ 8. 若无预测结果（cold start），改用训练集测试
if predictions.rdd.isEmpty():
    print("❗ No predictions generated. Using training set instead.")
    predictions = model.transform(training)
predictions.show()

# ✅ 9. 使用 RMSE 指标评估模型
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"📉 RMSE = {rmse:.3f}")

# ✅ 10. 为每位用户推荐 3 个商品
userRecs = model.recommendForAllUsers(3)
print("🔮 Top 3 Recommendations for Each User:")
userRecs.show(truncate=False)

# ✅ 在 userRecs 输出推荐后结束计时并打印用时
end_time = time.time()
print(f"⏱️ ALS 推荐流程耗时：{end_time - start_time:.2f} 秒")

# ✅ 1. 扁平化推荐结果
flatRecs = userRecs.withColumn("rec", explode("recommendations")) \
                   .select("userId", "rec.productId", "rec.rating")
                   
# ✅ 2. 显示扁平化后的推荐结果
flatRecs = flatRecs.filter("rating > 0")

# ✅ 2. 加载商品信息表（productId → productName）
productInfo = spark.read.csv("hdfs://localhost:9000/user/suzy/products_large.csv", header=True, inferSchema=True)

# ✅ 3. 联表，得到可读的推荐结果
finalRecs = flatRecs.join(productInfo, on="productId", how="left")

# ✅ 4. 显示推荐结果（含商品名）
print("🎁 Final Recommendation Results with Product Names:")
finalRecs.orderBy("userId", "rating", ascending=False).show(truncate=False)

# ✅ 5. 保存推荐结果到本地 CSV
finalRecs.coalesce(1).write.mode("overwrite").option("header", True).csv("recommendation_output")


# ✅ （可选）保存推荐结果到 HDFS
# finalRecs.coalesce(1).write.option("header", True).csv("hdfs://localhost:9000/user/suzy/recommendation_output")


# ✅ 11. 结束 Spark 会话
spark.stop()