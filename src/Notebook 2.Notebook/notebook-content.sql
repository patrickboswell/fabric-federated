-- Fabric notebook source

-- METADATA ********************

-- META {
-- META   "kernel_info": {
-- META     "name": "synapse_pyspark"
-- META   },
-- META   "dependencies": {
-- META     "lakehouse": {
-- META       "default_lakehouse": "1634bc81-9d9d-4494-b31a-5cf7e626fedd",
-- META       "default_lakehouse_name": "synthea",
-- META       "default_lakehouse_workspace_id": "86f20abc-c578-4508-9f8e-2e506a77a7dd"
-- META     }
-- META   }
-- META }

-- CELL ********************

-- Welcome to your new notebook
-- Type here in the cell editor to add code!


-- METADATA ********************

-- META {
-- META   "language": "sparksql",
-- META   "language_group": "synapse_pyspark"
-- META }

-- MARKDOWN ********************

-- select * from 

-- CELL ********************

SELECT * FROM synthea.kidney_disease LIMIT 1000

-- METADATA ********************

-- META {
-- META   "language": "sparksql",
-- META   "language_group": "synapse_pyspark"
-- META }
