# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "1634bc81-9d9d-4494-b31a-5cf7e626fedd",
# META       "default_lakehouse_name": "synthea",
# META       "default_lakehouse_workspace_id": "86f20abc-c578-4508-9f8e-2e506a77a7dd"
# META     }
# META   }
# META }

# CELL ********************

%pip install imblearn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow
from mlflow.models import infer_signature
from pyspark.sql.types import *
from pyspark.sql.functions import *
from ast import literal_eval

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT * FROM synthea.kidney_disease").toPandas()
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = df[df.LOS > 0.0].filter(
    items=[
        'LOS',
        'AGE',
        'DIABETES_FLAG',
        'INCOME_CAT',
        'GFR_MIN',
        'GFR_MAX',
        'PAYER_NAME'
    ]
)
df.DIABETES_FLAG = df.DIABETES_FLAG.astype(int)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

Y = df['DIABETES_FLAG']
for column, dtype in df.dtypes.items():
    if dtype == 'object':  # Assuming Decimal types are inferred as object
        try:
            # Attempt to convert each column to float
            df[column] = df[column].astype(float)
        except ValueError:
            # If conversion fails, it's not a Decimal column; move to next column
            df[column] = df[column].astype('category')
            continue
X = df.drop(columns='DIABETES_FLAG')
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Define the model hyperparameters
params = {
    "random_state": 8888,
}

# Assuming 'categorical_features' is a list of column names for categorical variables in your dataset
categorical_features = ['INCOME_CAT', 'PAYER_NAME']  # Replace with your actual column names

# Create a ColumnTransformer to apply OneHotEncoding to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'  # This leaves the rest of the columns unchanged
)

# Create a pipeline that first transforms the data, then oversamples using SMOTE, and finally fits the model
pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', HistGradientBoostingClassifier(**params))])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set using the pipeline
y_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a new MLflow Experiment
# Set given experiment as the active experiment. If an experiment with this name does not exist, a new experiment with this name is created.
mlflow.set_experiment("exp_kidney_disease_los")
mlflow.autolog(exclusive=False)

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model using the pipeline
    pipeline.fit(X_train, y_train)

    # Predict on the test set using the pipeline
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for synthe kidney data")

    # Infer the model signature
    signature = infer_signature(X_train, pipeline.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="kidney_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="kidney_model",
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
