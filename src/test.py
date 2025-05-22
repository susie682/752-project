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
x = 1000
# 读取用户和商品映射表
user_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_user_id") \
    .toDF("user_id", "userIndex")
item_ids = spark.read.option("header", "false").csv("hdfs://localhost:8020/user/ecommerce_project/mapping_asin_id") \
    .toDF("asin", "itemIndex")

# 读取评分数据（使用小数据集）
ratings = spark.read.option("header", "false").csv(f"hdfs://localhost:8020/user/ecommerce_project/cleaned_data_{x}") \
    .toDF("user_id", "asin", "rating")
ratings = ratings.withColumn("rating", col("rating").cast("float"))

# 缓存并输出调试信息
ratings.cache()



# 加入索引，构造 ALS 输入格式
indexed = ratings.join(user_ids, on="user_id", how="inner") \
                 .join(item_ids, on="asin", how="inner") \
                 .select(col("userIndex").cast("int"), col("itemIndex").cast("int"), col("rating"))
start_time = time.time()
# 拆分训练 / 测试集

training, test = indexed.randomSplit([0.8, 0.2], seed=42)
# 构建并训练 ALS 模型
als = ALS(
    maxIter=15,
    regParam=0.05,
    rank=20,
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



# 评估 RMSE
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Data Count =  {x}")
print(f"📉ALS  RMSE = {rmse:.3f}")


# 推荐流程计时结束
end_time = time.time()
print(f"⏱️ ALS recommendation time cost：{end_time - start_time:.2f} Seconds")

print(f"数据总量: {ratings.count()}")


# 停止 Spark
spark.stop()