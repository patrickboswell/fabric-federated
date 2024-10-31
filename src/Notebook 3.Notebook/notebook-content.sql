-- Fabric notebook source

-- METADATA ********************

-- META {
-- META   "kernel_info": {
-- META     "name": "synapse_pyspark"
-- META   },
-- META   "dependencies": {
-- META     "lakehouse": {
-- META       "default_lakehouse": "929e4e79-cbe4-436c-a614-5f44fa50f496",
-- META       "default_lakehouse_name": "NPRQI",
-- META       "default_lakehouse_workspace_id": "f72f59d8-9015-4de8-bab1-573ef077c941",
-- META       "known_lakehouses": [
-- META         {
-- META           "id": "929e4e79-cbe4-436c-a614-5f44fa50f496"
-- META         }
-- META       ]
-- META     }
-- META   }
-- META }

-- CELL ********************


SELECT * 
FROM NPRQI.nprqi_homebrew_project_info 
LIMIT 1000
;

-- METADATA ********************

-- META {
-- META   "language": "sparksql",
-- META   "language_group": "synapse_pyspark"
-- META }
