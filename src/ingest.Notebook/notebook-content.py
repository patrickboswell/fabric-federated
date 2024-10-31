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
# META       "default_lakehouse_workspace_id": "f72f59d8-9015-4de8-bab1-573ef077c941",
# META       "known_lakehouses": [
# META         {
# META           "id": "acccfb51-5a60-4be1-b423-97f71aa4edec"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************


# CELL ********************

diabetes_dataset_url = "https://utexas-my.sharepoint.com/:x:/g/personal/damien_johnson_austin_utexas_edu/Ea4OS4J2wy9Dndq9jPPp5KAB8NXa0UWavKHQeFvfINsFxg?e=6Fu6v5"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from notebookutils import mssparkutils
import requests

#create subfolder
mssparkutils.fs.mkdirs("Files/diabetesdataset")

#download the CSV file from OneDrive and save to the folder
with requests.Session() as s:
    download = s.get(diabetes_dataset_url)
    #print(download.content.decode())
    mssparkutils.fs.put("Files/diabetesdataset/diabetes.csv", download.content.decode(), True)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Read the Diabetes Dataset file
df = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("Files/diabetesdataset/diabetes.csv")
# df now is a Spark DataFrame containing CSV data from "Files/diabetestdataset/diabetes.csv".
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#print schema
df.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


df = df.withColumnRenamed("plasma glucose","plasma_glucose").withColumnRenamed("blood pressure","blood_pressure") \
.withColumnRenamed("triceps skin thickness", "triceps_skin_thickness") \
.withColumnRenamed("diabetes pedigree", "diabetes_pedigree")   

display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# VertiParquet writer optimizes the Delta Lake parquet files resulting in 3x-4x compression improvement and up to 10x performance acceleration over Delta Lake files not optimized using VertiParquet while still maintaining full Delta Lake and PARQUET format compliance.
# 
# optimizeWrite It dynamically optimizes files during write operations generating files with a default 128 MB size. The target file size may be changed per workload requirements using configurations.

# CELL ********************

spark.conf.set("sprk.sql.parquet.vorder.enabled", "true") # Enable Verti-Parquet write
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true") # Enable automatic delta optimized write

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Write Spark dataframe to lakehouse delta table

# CELL ********************

table_name = "diabetes"
df.write.mode("overwrite").format("delta").save(f"Tables/{table_name}")
print(f"Spark dataframe saved to delta table: {table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
