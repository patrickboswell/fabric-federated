# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "226b6e08-8559-4dc8-844e-7497c7c45cdc",
# META       "default_lakehouse_name": "lakehousefranksynthea",
# META       "default_lakehouse_workspace_id": "f72f59d8-9015-4de8-bab1-573ef077c941"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Track Machine Learning experiments and models
# Machine learning model that predicts encounter duration, detect anomalies, and provides answers to time-related statistics questions such as how much time an encounter takes in average? what is the maximum and minimum time for an encounter?
# 
# First, compute the duration of each encounter using the START and STOP columns. This can be achieved by converting them into a timestamp format and calculating the difference.

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp

# Create Spark session
spark = SparkSession.builder.appName("strokeencountertimeanalysis").getOrCreate()

# Load table
df = spark.table("synthea_stroke_claims_transactions")

# Convert START and STOP to timestamp and calculate duration
df = df.withColumn("START_TS", unix_timestamp(col("START"), "yyyy-MM-dd HH:mm:ss").cast("timestamp")) \
       .withColumn("STOP_TS", unix_timestamp(col("STOP"), "yyyy-MM-dd HH:mm:ss").cast("timestamp")) \
       .withColumn("DURATION", (col("STOP_TS").cast("long") - col("START_TS").cast("long")) / 60)  # Duration in minutes

df.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# 
# **Analyzing Duration Statistics**
# 
# <br>
# Calculate average, maximum, and minimum encounter durations.

# CELL ********************

from pyspark.sql import functions as F

# Calculate average, maximum, and minimum duration
stats = df.agg(
    F.avg("DURATION").alias("avg_duration"),
    F.max("DURATION").alias("max_duration"),
    F.min("DURATION").alias("min_duration")
).collect()

# Check if there are any results
if stats:
    avg_duration = stats[0]["avg_duration"]
    max_duration = stats[0]["max_duration"]
    min_duration = stats[0]["min_duration"]

    print(f"Average Duration: {avg_duration}, Max Duration: {max_duration}, Min Duration: {min_duration}")
else:
    print("No results found.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Prediction of Encounter Durations**

# CELL ********************

from pyspark.sql.functions import col, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Feature Engineering ( extract day of week from START)
df = df.withColumn("DAY_OF_WEEK", dayofweek(col("START_TS")))

# Prepare features and label
assembler = VectorAssembler(inputCols=["DAY_OF_WEEK"], outputCol="features")
train_data = assembler.transform(df).select("features", "DURATION")

# Train a linear regression model
lr = LinearRegression(featuresCol="features", labelCol="DURATION")
model = lr.fit(train_data)

# Make predictions
predictions = model.transform(train_data)
predictions.select("features", "DURATION", "prediction").show(n=len(predictions.collect()), truncate = False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Calculate errors
predictions_pd['error'] = predictions_pd['prediction'] - predictions_pd['DURATION']

plt.figure(figsize=(10, 6))
plt.hist(predictions_pd['error'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (minutes)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Calculate errors in the DataFrame
predictions_with_errors = predictions.withColumn("error", F.col("prediction") - F.col("DURATION"))

# Filter high errors
high_error_threshold = 40  # can adjust threshold as needed
high_errors_df = predictions_with_errors.filter(F.abs(F.col("error")) > high_error_threshold)

# Show the results
high_errors_df.select("features", "DURATION", "prediction", "error").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Detecting Anomalies**

# CELL ********************

from pyspark.sql import functions as F

# Calculate average and standard deviation of DURATION
stats = df.agg(
    F.avg("DURATION").alias("avg_duration"),
    F.stddev("DURATION").alias("stddev_duration")
).collect()

# Check if stats has results
if stats:
    # Extract mean and standard deviation
    mean_duration = stats[0]["avg_duration"]
    stddev_duration = stats[0]["stddev_duration"]

    # Calculate Z-scores for DURATION
    df = df.withColumn("Z_SCORE", (col("DURATION") - mean_duration) / stddev_duration)

    # Define anomaly as a Z-score above a threshold (e.g., 3)
    df = df.withColumn("ANOMALY", F.when(F.abs(col("Z_SCORE")) > 3, 1).otherwise(0))

    # Show anomalies
    df.filter(col("ANOMALY") == 1).show()

    # Create a DataFrame with anomalies
    df_with_anomalies = df.filter(col("ANOMALY") == 1)

    # Show the DataFrame with anomalies in a structured format
    df_with_anomalies.show(truncate=False)  # Set truncate=False to see full column values

    # Convert to Pandas DataFrame
    pandas_df = df_with_anomalies.toPandas()

    # Display using Pandas
    import pandas as pd
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Prevent line breaks
    print(pandas_df)

    # Print each row formatted
    for row in df_with_anomalies.collect():
        print(f"Encounter ID: {row['ENCOUNTER_ID']}, Duration: {row['DURATION']}, Anomaly: {row['ANOMALY']}")
else:
    print("No results found for duration statistics.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
