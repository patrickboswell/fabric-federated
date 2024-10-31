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

# Welcome to your new notebook
# Type here in the cell editor to add code!
#!/usr/bin/env python

import pandas as pd 
import requests
import json 
from pyspark.sql import *
import pyspark 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

data = {
    'token': '4F22F7674043A3CABC7D9E99F2BD9026',
    'content': 'project',
    'format': 'json',
    'returnFormat': 'json'
}
r = requests.post('https://redcap.dellmed.utexas.edu/api/',data=data)

print('HTTP Status: ' + str(r.status_code))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Parse the JSON response into a dictionary
json_data = r.json()

# Create a Spark DataFrame from the list of dictionaries
df = spark.createDataFrame([json_data])



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Write the DataFrame to a table in the Lakehouse
df.write.mode("overwrite").saveAsTable("nprqi_homebrew_project_info")

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
