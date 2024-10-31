# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "acccfb51-5a60-4be1-b423-97f71aa4edec",
# META       "default_lakehouse_name": "diabetes",
# META       "default_lakehouse_workspace_id": "f72f59d8-9015-4de8-bab1-573ef077c941"
# META     }
# META   }
# META }

# MARKDOWN ********************

# ### Train the machine learning model to predict the likelihood of an individual getting diabetes based on historical diagnostic measurements available in the dataset.

# CELL ********************

import mlflow
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from synapse.ml.train import ComputeModelStatistics

# Set experiment name
EXPERIMENT_NAME = "diabetes_high_risk"
mlflow.set_experiment(EXPERIMENT_NAME)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load and preprocess data
data_df = spark.read.format("delta").load("Tables/diabetes_refined")
data_df = data_df.drop("insulin", "bmi")

display(data_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### splitting a dataset into two parts: a training set and a test set

# CELL ********************

train_test_split = [0.75, 0.25]
seed = 1234
train_df, test_df = data_df.randomSplit(train_test_split, seed=seed)

# Print record counts
print(f"Train set record count: {train_df.count()}")
print(f"Test set record count: {test_df.count()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### organizing the columns of the training dataset into two categories: numeric and categorical.

# CELL ********************

# Identify numeric and categorical columns
numeric_cols = train_df.drop("diabetes", "bmi", "insulin", "insulin_level", "obesity_level").columns
categorical_cols = ["insulin_level", "obesity_level"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### setting up a data preprocessing pipeline for a machine learning model.

# CELL ********************

# Initialize stages for the pipeline
stages = []

# One-hot encode categorical columns
for col in categorical_cols:
    index_col = col + "_index"
    vec_col = col + "_vec"
    stringIndexer = StringIndexer(inputCol=col, outputCol=index_col)
    stages.append(stringIndexer)
    encoder = OneHotEncoder(inputCols=[index_col], outputCols=[vec_col])
    stages.append(encoder)

# Use VectorAssembler to generate feature column for the ML model
assemblerInputs = numeric_cols + [vec_col for col in categorical_cols]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features", handleInvalid="skip")
stages.append(assembler)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### configuring a Random Forest classifier, which is a type of machine learning model used for classification tasks

# CELL ********************

# Configure RandomForestClassifier
rf = RandomForestClassifier(
    labelCol="diabetes",
    featuresCol="features",
    numTrees=100
)

# Add the classifier to the stages
stages.append(rf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#  ### managing an MLflow run, which is a tool for tracking machine learning experiments

# CELL ********************

# Start MLflow run to capture parameters, metrics, and log the model
with mlflow.start_run():
    # Define the pipeline
    ml_pipeline = Pipeline(stages=stages)

    # Log the parameters used in training for tracking purposes
    mlflow.log_param("train_test_split", train_test_split)
    mlflow.log_param("numTrees", 100)

    # Call fit method on the pipeline with training subset data to create ML Model
    rf_model = ml_pipeline.fit(train_df)

    # Perform predictions on the test subset of the data
    rf_predictions = rf_model.transform(test_df)

    # Measure and log metrics to track performance of the model
    metrics = ComputeModelStatistics(
        evaluationMetric="classification",
        labelCol="diabetes",
        scoredLabelsCol="prediction"
    ).transform(rf_predictions)

    mlflow.log_metric("precision", round(metrics.first()["precision"], 4))
    mlflow.log_metric("recall", round(metrics.first()["recall"], 4))
    mlflow.log_metric("accuracy", round(metrics.first()["accuracy"], 4))   
    mlflow.log_metric("AUC", round(metrics.first()["AUC"], 4))

    # Log the model for subsequent use
    model_name = "diabetes-rf"
    mlflow.spark.log_model(rf_model, artifact_path=model_name, registered_model_name=model_name, dfs_tmpdir="Files/tmp/mlflow")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
