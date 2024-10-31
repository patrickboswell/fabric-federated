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

# ### Simulate input diabetes diagnostic data to be used for predictions using faker


# CELL ********************

%pip install Faker==18.10.1


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

diabDataSchema = StructType(
[
    StructField('pregnancies', IntegerType(), True),
    StructField('plasma_glucose', IntegerType(), True),
    StructField('blood_pressure', IntegerType(), True),
    StructField('triceps_skin_thickness', IntegerType(), True),
    StructField('insulin_level', StringType(), True),
    StructField('obesity_level', StringType(), True),
    StructField('diabetes_pedigree', DoubleType(), True),
    StructField('age', IntegerType(), True)
]
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### generates fake data to perform batch predictions

# CELL ********************

from faker import Faker

faker = Faker()
simulateRecordCount = 10
simData = []

for i in range(simulateRecordCount):
    pregnancies = faker.random_int(0,8)
    plasma_glucose = faker.random_int(70, 170)
    blood_pressure = faker.random_int(50, 120)
    triceps_skin_thickness = faker.random_int(10, 50)    
    diabetes_pedigree = faker.pyfloat(right_digits = 3, positive = True, max_value = 2.42)
    age = faker.random_int(21, 81)

    insulin_level = faker.random_element(elements=('normal','abnormal'))
    obesity_level = faker.random_element(elements=('underweight','normal','overweight','obese'))


    simData.append((pregnancies, plasma_glucose, blood_pressure, triceps_skin_thickness, insulin_level, obesity_level, diabetes_pedigree, age))

#print(simData)

df = spark.createDataFrame(data = simData, schema = diabDataSchema)
display(df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Import libraries, load the models, make predictions, display predictions in format suitable for storage

# CELL ********************

import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from synapse.ml.core.platform import *
from synapse.ml.lightgbm import LightGBMRegressor

model_uri = "models:/diabetes-rf/latest"
model = mlflow.spark.load_model(model_uri)

predictions_df = model.transform(df)
display(predictions_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### format predictions for consumption 

# CELL ********************

from pyspark.sql.functions import get_json_object
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.functions import format_number

firstelement=udf(lambda v: float(v[0]) if (float(v[0]) >  float(v[1])) else float(v[1]), FloatType())

predictions_formatted_df = predictions_df \
    .withColumn("prob", format_number(firstelement('probability'), 4)) \
    .withColumn("diab_pred", predictions_df.prediction.cast('int')) \
    .drop("features", "rawPrediction", "probability", "prediction", "insulin_level_vec", "obesity_level_vec")

display(predictions_formatted_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# write prediction results to new table

# CELL ********************

# optimize writes to Delta Table
spark.conf.set("sprk.sql.parquet.vorder.enabled", "true") # Enable Verti-Parquet write
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true") # Enable automatic delta optimized write

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

table_name = "diabetes_predictions"
predictions_formatted_df.write.mode("overwrite").format("delta").save(f"Tables/{table_name}")
print(f"Output Predictions saved to delta table: {table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAGIC %%sql
# MAGIC select * from diabetes_predictions limit 10;

# METADATA ********************

# META {
# META   "language": "sparksql",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ###  inspect the history and metadata of a Delta table in a Spark environment.

# CELL ********************

from delta.tables import DeltaTable
delta_table = DeltaTable.forPath(spark, f"Tables/{table_name}")
delta_table.history().show()
delta_table.detail().show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

