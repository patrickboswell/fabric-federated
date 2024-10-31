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

# Hello and Welcome 
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡤⠶⠒⠒⠒⠒⠲⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠖⠉⠁⠀⠀⠀⠀⠁⠀⠐⠤⣄⠉⠓⢦⣄⠀⠀⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠣⣀⠀⠈⠳⣄⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠏⠈⠁⠀⠀⠉⠀⠀⠀⠁⠉⠉⠁⠀⠀⢀⡠⠀⣄⠀⠀⠘⢧⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⢰⠇⢠⠀⠀⠀⠒⠀⠈⠒⠀⠀⠀⠠⠤⠆⠀⠁⠀⠀⠀⠁⠀⢆⠈⣇
#⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⠈⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⢸
#⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⢳⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢀⣸
#⠀⠀⠀⠀⠀⠀⠀⠀⠘⣟⣼⡆⠀⠀⠀⠀⣀⡀⠠⠤⢄⣀⠤⠄⠀⠀⠀⠀⠀⠀⠀⡎⣞⡏
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⢿⣇⣤⠄⠀⠀⢀⡉⠉⠓⠒⠒⠒⠋⠉⡁⠀⠀⠠⢤⣀⣿⢿⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⣉⡤⢤⢤⢤⣀⡉⠲⡀⠀⠀⠀⡴⠊⣁⡠⡤⡤⠤⣈⡻⡟⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠁⣿⣿⠛⠛⢻⢷⣯⡂⠘⠀⠀⠚⢐⣮⠿⠿⣟⣿⣿⣿⡎⣿⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣦⡘⣿⣿⣷⣿⣿⣿⣿⡄⠀⠀⢀⣾⣷⣤⣶⣿⣿⣿⡏⣠⡏⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣵⡙⢯⣻⢿⣿⣿⣿⣿⠀⠀⢸⢿⣿⣿⡿⣟⡿⢋⣼⠟⠀⠀
#⢀⣠⣀⠀⠀⢠⠖⢲⡀⠀⠀⠈⢻⣄⠈⠉⠿⠿⠛⠋⠀⠀⠘⠛⠛⠿⠏⠉⢀⡽⠋⠀⠀⠀
#⢸⡅⠹⡄⠀⢸⡄⢸⠇⠀⠀⠀⠀⢻⣑⡀⠀⠀⠀⠾⠖⠰⡾⡆⠀⠀⠀⣜⡽⠁⠀⠀⠀⠀
#⠀⢳⡄⢷⠀⢰⠃⢻⠀⠀⠀⠀⠀⠀⠻⡷⣄⡀⠀⠀⠀⠀⠀⠀⠀⣠⣼⠟⠁⠀⠀⠀⠀⠀
#⠀⠈⢧⡈⣧⢸⡀⣼⠀⠀⠀⠀⠀⠀⠀⠈⠻⣧⠈⠉⠭⠭⠍⢁⢰⡿⠋⠀⠀⠀⠀⠀⠀⠀
#⠀⠀⠘⣇⠈⡿⠀⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣷⣄⠀⠁⢀⣼⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀
#⠀⡴⢺⠯⠤⠤⠖⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⡷⢶⣿⢿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
#⠀⠻⡞⠳⢶⣾⡶⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⣼⢿⠁⠀⡝⣾⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
#⠀⠀⠙⢦⠴⠃⠀⠀⣸⠀⠀⠀⢀⣀⣤⣠⠤⠾⢁⡀⣇⢠⠇⡅⠻⠦⣄⣀⣀⡀⠀⠀⠀⠀
#⠀⠀⠀⠘⢷⣜⠒⠲⣏⠀⠀⡴⡿⠀⠀⠈⠁⠒⠋⠀⠈⠉⠀⠘⠒⠖⠁⠀⠀⠩⣳⡀⠀⠀
#⠀⠀⠀⠀⠀⢹⡐⢧⢻⠀⠀⣷⠁⠀⢰⠀⠀⠀⠀⠀⠲⡶⠂⠀⠀⠀⠀⡇⠀⠀⢧⡇⠀⠀
#⠀⠀⠀⠀⠀⠈⡇⠀⠈⣧⠀⣟⠀⠀⣨⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⣸⠀⠀⢩⡇⠀⠀
#⠀⠀⠀⠀⠀⠀⢻⠀⠀⠘⡾⠁⠐⣾⡟⡷⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢾⣿⠁⠀⠉⡇⠀⠀
#⠀⠀⠀⠀⠀⠀⠸⣆⠀⠀⠸⡀⢰⣿⠁⢿⢄⡀⠀⠀⠴⠷⠄⠀⠀⢀⣾⡏⠀⠀⠀⡇⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⣯⣆⠀⠀⢹⣟⡏⠀⢸⠦⠀⠀⠀⠀⡄⠀⠀⠀⠠⣧⣿⠀⠀⢸⡇⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠸⣜⢦⣀⡜⡞⠀⠀⢸⡓⢀⡁⠀⠀⡇⠀⠀⡀⠐⣿⡇⠀⠀⠈⡇⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠙⢶⣥⡽⠁⠀⠀⢸⣅⣤⣄⣀⣀⣀⣀⣀⣤⣈⡿⢷⡀⣀⣸⠇⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠀⠀⠀ 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import plotly as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from statsmodels.formula.api import logit
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# import pymc as pm
# import arviz as az
# import seaborn as sns
# from eli5 import show_weights
# from eli5 import show_prediction
# import plotly.express as px
# from numpy import random
# from IPython.display import HTML

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT * FROM synthea.kidney_disease LIMIT 10")
display(df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df[sorted(df.columns)].dtypes

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df.columns

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

edf_flag_dummies = pd.get_dummies(df['ed_flag'], prefix='ed_flag', dtype=float)
patient_age_dummies = pd.get_dummies(df['age'], prefix='patient_age', dtype=float)
patient_payer_dummies = pd.get_dummies(df['payer_name'], prefix='patient_payer', dtype=float)
# gfr_min_dummies = pd.get_dummies(df['gfr_min'], prefix='gfr_min', dtype=float)
# gfr_max_dummies = pd.get_dummies(df['gfr_max'], prefix='gfr_max', dtype=float)
income_cat_dummies = pd.get_dummies(df['income_cat'], prefix='income_cat', dtype=float)

df = pd.concat([df, edf_flag_dummies, patient_age_dummies, income_cat_dummies], axis=1)

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
