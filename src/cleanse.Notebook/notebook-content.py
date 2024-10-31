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

# CELL ********************

data_df = spark.read.format("delta").load("Tables/diabetes")
display(data_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT * FROM diabetes_refined")

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

# CELL ********************

data_df.dtypes

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(data_df.summary())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# extracting summary statistics for the "age" column 
display(data_df.select("age").summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(data_df.groupBy("age").count())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# CELL ********************

#creates a new DataFrame where all 0 values in the specified columns are replaced with None
data_df_fillna = data_df.replace(0,None,['plasma_glucose','blood_pressure','triceps_skin_thickness','insulin','bmi'])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#check and display all the rows in the DataFrame where the triceps_skin_thickness value is missing
display(data_df_fillna.filter("triceps_skin_thickness IS NULL"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# displays a count of non-null values for each column
data_df_fillna.summary("count").show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#assign values to the obesity_level column based on 'patient's weight
from pyspark.sql.functions import when

data_df_newbmi = data_df_fillna.withColumn('obesity_level', when(data_df_fillna.bmi <= 18.5, 'underweight')
                    .when((data_df_fillna.bmi > 18.5) & (data_df_fillna.bmi <= 24.9), 'normal')
                    .when((data_df_fillna.bmi > 24.9) & (data_df_fillna.bmi <= 29.9), 'overweight')
                    .otherwise('obese'))

display(data_df_newbmi)  

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#assign values to new column insulin_level based on insulin level (numeric)
data_df_refined = data_df_newbmi.withColumn('insulin_level', 
              when(data_df_newbmi.insulin <= 16, 'normal')
              .otherwise('abnormal'))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(data_df_refined)


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

table_name = "diabetes_refined"
data_df_processed.write.mode("overwrite").format("delta").save(f"Tables/{table_name}")
print(f"Spark dataframe saved to delta table: {table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAGIC %%sql
# MAGIC 
# MAGIC select * from diabetes_refined limit 10;

# METADATA ********************

# META {
# META   "language": "sparksql",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAGIC %%sql 
# MAGIC 
# MAGIC select obesity_level, diabetes, count(*) as count
# MAGIC from diabetes_refined 
# MAGIC --where diabetes = 1
# MAGIC GROUP By obesity_level, diabetes
# MAGIC      

# METADATA ********************

# META {
# META   "language": "sparksql",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
