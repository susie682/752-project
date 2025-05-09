from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.mllib.recommendation import Rating, MatrixFactorizationModel, ALS as SVD_ALS
import time

# ✅ 1. 创建 SparkSession 并连接 HDFS
spark = SparkSession.builder \
    .appName("SVD_ECommerce_Recommendation") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

sc = spark.sparkContext

# ✅ 2. 加载评分数据
ratings_df = spark.read.csv("hdfs://localhost:9000/user/suzy/ratings_large.csv", header=True, inferSchema=True)
ratings_df.printSchema()
ratings_df.show()

# ✅ 3. 开始计时
start_time = time.time()

# ✅ 4. 转换为 RDD[Rating]
ratings_rdd = ratings_df.select("userId", "productId", "rating") \
    .rdd.map(lambda row: Rating(int(row["userId"]), int(row["productId"]), float(row["rating"])))

# ✅ 5. 拆分训练集 / 测试集
training_rdd, test_rdd = ratings_rdd.randomSplit([0.8, 0.2], seed=42)

# ✅ 6. 训练 SVD ALS 模型
model = SVD_ALS.train(training_rdd, rank=10, iterations=10, lambda_=0.1)

# ✅ 7. 如果测试集为空则使用训练集
test_user_product = test_rdd.map(lambda r: (r[0], r[1]))
if test_user_product.isEmpty():  # 🔧 添加的容错处理
    print("❗ No test data. Using training set instead.")
    test_user_product = training_rdd.map(lambda r: (r[0], r[1]))
    rates = training_rdd.map(lambda r: ((r[0], r[1]), r[2]))
else:
    rates = test_rdd.map(lambda r: ((r[0], r[1]), r[2]))

# ✅ 8. 预测评分 + 计算 RMSE
predictions = model.predictAll(test_user_product).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = rates.join(predictions)
mse = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
rmse = mse ** 0.5
print(f"📉 RMSE = {rmse:.3f}")

# ✅ 9. 为每位用户推荐 3 个产品
userRecs = model.recommendProductsForUsers(3)

# ✅ 10. 计时结束
end_time = time.time()
print(f"⏱️ SVD 推荐流程耗时：{end_time - start_time:.2f} 秒")

# ✅ 11. 展平推荐结果为 DataFrame
flatRecs = userRecs.flatMapValues(lambda recs: recs) \
    .map(lambda x: (x[0], x[1].product, x[1].rating)) \
    .toDF(["userId", "productId", "rating"])
flatRecs = flatRecs.filter("rating > 0")

# ✅ 12. 加载商品信息表（productId → productName）
productInfo = spark.read.csv("hdfs://localhost:9000/user/suzy/products_large.csv", header=True, inferSchema=True)
finalRecs = flatRecs.join(productInfo, on="productId", how="left")

# ✅ 13. 显示推荐结果（含商品名）
print("🎁 Final Recommendation Results with Product Names:")
finalRecs.orderBy("userId", "rating", ascending=False).show(truncate=False)

# ✅ 14. 保存推荐结果到本地 CSV
finalRecs.coalesce(1).write.mode("overwrite").option("header", True).csv("svd_recommendation_output")

# ✅ 15. 结束 Spark 会话
spark.stop()
