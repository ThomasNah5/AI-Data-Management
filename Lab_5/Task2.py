from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, spark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col

# Re-read the data for a fresh start
df2 = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("titanic.csv")

# Select relevant columns

dataset2 = df2.select(
    col('Survived').cast('float'),
    col('Pclass').cast('float'),
    col('Sex'),
    col('Age').cast('float'),
    col('Fare').cast('float'),
    col('Embarked')
)

# Clean the data
dataset2 = dataset2.replace('?', None).dropna(how='any')

print(f"\nRecords after cleaning: {dataset2.count()}")
dataset2.show(5)



# Define pipeline stages
sex_indexer = StringIndexer(
    inputCol='Sex',
    outputCol='Gender',
    handleInvalid='keep'
)


embarked_indexer = StringIndexer(
    inputCol='Embarked',
    outputCol='Boarded',
    handleInvalid='keep'
)

feature_cols = ['Pclass', 'Age', 'Fare', 'Gender', 'Boarded']
assembler2 = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)

rf2 = RandomForestClassifier(
    labelCol='Survived',
    featuresCol='features',
    maxDepth=5,
    numTrees=20
)

# Create the Pipeline
pipeline = Pipeline(stages=[
    sex_indexer,
    embarked_indexer,
    assembler2,
    rf2
])

print("\n Pipeline Stages ")
for i, stage in enumerate(pipeline.getStages()):
    print(f"Stage {i+1}: {type(stage).__name__}")

# Split data
(train_data2, test_data2) = dataset2.randomSplit([0.8, 0.2], seed=42)
print(f"\nTraining samples: {train_data2.count()}")
print(f"Test samples: {test_data2.count()}")

# Fit the pipeline (trains the entire pipeline at once)
pipeline_model = pipeline.fit(train_data2)

# Make predictions
predictions2 = pipeline_model.transform(test_data2)

print("\n Pipeline Predictions ")
predictions2.select('Survived', 'prediction', 'features').show(10)

# Evaluate the pipeline model
evaluator2 = MulticlassClassificationEvaluator(
    labelCol='Survived',
    predictionCol='prediction',
    metricName='accuracy'
)

accuracy2 = evaluator2.evaluate(predictions2)


# Stop Spark session
spark.stop()
print("\nSpark session stopped.")
