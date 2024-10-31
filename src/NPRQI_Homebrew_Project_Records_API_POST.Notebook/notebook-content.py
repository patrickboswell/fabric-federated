# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "929e4e79-cbe4-436c-a614-5f44fa50f496",
# META       "default_lakehouse_name": "NPRQI",
# META       "default_lakehouse_workspace_id": "f72f59d8-9015-4de8-bab1-573ef077c941"
# META     }
# META   }
# META }

# CELL ********************

# This notebook makes an API call to REDCap and writes the fetched data to a table in the NPRQI lakehouse
#!/usr/bin/env python

import pandas as pd 
import requests
import json 
from pyspark.sql import *
import pyspark 
from datetime import datetime
from delta.tables import DeltaTable
from pyspark.sql.functions import lit
from pyspark.sql import functions as F



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


data = {
    'token': '4F22F7674043A3CABC7D9E99F2BD9026',
    'content': 'record',
    'action': 'export',
    'format': 'json',
    'type': 'eav',
    'csvDelimiter': '',
    'rawOrLabel': 'label',
    'rawOrLabelHeaders': 'label',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'json'
}

# Make POST request
r = requests.post('https://redcap.dellmed.utexas.edu/api/',data=data)
print('HTTP Status: ' + str(r.status_code))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Read data as JSON
json_data = r.json()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# Create a Spark DataFrame
api_df = spark.createDataFrame(data=json_data)

# Capture the current datetime when the API call is made
export_datetime = datetime.now()

# Add the export datetime as a new column
api_df = api_df.withColumn("export_datetime", lit(export_datetime))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

max_value = api_df.agg(F.max(F.col("record").cast("int"))).collect()[0][0]
print(max_value)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Read existing table as df 
table_df = spark.sql("select * from nprqi_homebrew_project_records")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Left Anti Join (records in df2 that do not have a match in df1)
api_df_unmatched = api_df.join(table_df, on="record", how="left_anti")

# Display the unmatched records
api_df_unmatched.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Append unmatched data to the table in lakehouse 
api_df_unmatched.write.format("delta").mode("append").saveAsTable('nprqi_homebrew_project_records')

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


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
