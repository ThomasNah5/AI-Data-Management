from pyspark.sql import SparkSession

# Re-read the data for a fresh start
df2 = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("titanic.csv")

# Select relevant columns
from pyspark.sql.functions import col

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

# Import Pipeline and required components
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, spark
from pyspark.ml.classification import RandomForestClassifier

# Define pipeline stages
# Stage 1: Index 'Sex' column
sex_indexer = StringIndexer(
    inputCol='Sex',
    outputCol='Gender',
    handleInvalid='keep'
)

# Stage 2: Index 'Embarked' column
embarked_indexer = StringIndexer(
    inputCol='Embarked',
    outputCol='Boarded',
    handleInvalid='keep'
)

# Stage 3: Assemble features into a vector
feature_cols = ['Pclass', 'Age', 'Fare', 'Gender', 'Boarded']
assembler2 = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)

# Stage 4: Random Forest Classifier
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


# Additional metrics for Pipeline model
print(f"Precision: {precision_evaluator.evaluate(predictions2):.4f}")
print(f"Recall: {recall_evaluator.evaluate(predictions2):.4f}")
print(f"F1 Score: {f1_evaluator.evaluate(predictions2):.4f}")


print("\n\n" + "="*70)
print("COMPARISON: Project 1 vs Project 2")
print("="*70)
print(f"Project 1 (Without Pipeline) Accuracy: {accuracy2:.4f}")
print(f"Project 2 (With Pipeline) Accuracy:    {accuracy2:.4f}")
print("\nKey Differences:")
print("- Project 1: Manual sequential preprocessing steps")
print("- Project 2: All transformations chained in a Pipeline")
print("\nAdvantages of Pipeline approach:")
print("1. Cleaner, more maintainable code")
print("2. Easy to save and load the entire workflow")
print("3. Prevents data leakage during cross-validation")
print("4. Simplifies hyperparameter tuning with CrossValidator")

# Stop Spark session
spark.stop()
print("\nSpark session stopped.")
