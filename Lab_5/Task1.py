from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import isnull, when, count
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



spark = SparkSession \
    .builder \
    .appName('Titanic ML - Without Pipeline') \
    .getOrCreate()

print("SparkSession created successfully!")
print(spark)

df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("titanic.csv")

print("\n Original Dataset ")
df.show(5)
print(f"Total records: {df.count()}")
df.printSchema()




dataset = df.select(
    col('Survived').cast('float'),
    col('Pclass').cast('float'),
    col('Sex'),
    col('Age').cast('float'),
    col('Fare').cast('float'),
    col('Embarked')
)

print("\n Selected Columns ")
dataset.show(5)



print("\n Null Value Count (Before Cleaning) ")
dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()


dataset = dataset.replace('?', None) \
    .dropna(how='any')

print("\n Null Value Count (After Cleaning) ")
dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()
print(f"Records after cleaning: {dataset.count()}")



# Convert 'Sex' to 'Gender' (numeric)
dataset = StringIndexer(
    inputCol='Sex',
    outputCol='Gender',
    handleInvalid='keep'
).fit(dataset).transform(dataset)

# Convert 'Embarked' to 'Boarded' (numeric)
dataset = StringIndexer(
    inputCol='Embarked',
    outputCol='Boarded',
    handleInvalid='keep'
).fit(dataset).transform(dataset)

print("\n After String Indexing")
dataset.show(5)


dataset = dataset.drop('Sex')
dataset = dataset.drop('Embarked')

print("\nAfter Dropping Original Categorical Columns ")
dataset.show(5)



required_features = ['Pclass', 'Age', 'Fare', 'Gender', 'Boarded']
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(dataset)

print("\n After Vector Assembly ")
transformed_data.show(5)


(training_data, test_data) = transformed_data.randomSplit([0.8, 0.2], seed=42)
print(f"\nNumber of training samples: {training_data.count()}")
print(f"Number of test samples: {test_data.count()}")

# Train RandomForest Classifier


rf = RandomForestClassifier(
    labelCol='Survived',
    featuresCol='features',
    maxDepth=5
)

model = rf.fit(training_data)

# Make predictions
predictions = model.transform(test_data)

print("\n Predictions")
predictions.select('Survived', 'prediction', 'features').show(10)

#Evaluate the model


evaluator = MulticlassClassificationEvaluator(
    labelCol='Survived',
    predictionCol='prediction',
    metricName='accuracy'
)

accuracy = evaluator.evaluate(predictions)


# Additional metrics
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol='Survived', predictionCol='prediction', metricName='weightedPrecision')
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol='Survived', predictionCol='prediction', metricName='weightedRecall')
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol='Survived', predictionCol='prediction', metricName='f1')

print(f"Precision: {precision_evaluator.evaluate(predictions):.4f}")
print(f"Recall: {recall_evaluator.evaluate(predictions):.4f}")
print(f"F1 Score: {f1_evaluator.evaluate(predictions):.4f}")



