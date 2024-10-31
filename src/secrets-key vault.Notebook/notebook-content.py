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

# CELL ********************

from mssparkutils.credentials import getSecret

# Replace with your Key Vault name and secret name
key_vault_name = "your_key_vault_name"
secret_name = "your_secret_name"

secret_value = getSecret(f"https://{key_vault_name}.vault.azure.net/", secret_name)
print(secret_value)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import requests

client_id = getSecret(f"https://{key_vault_name}.vault.azure.net/", "client_id")
client_secret = getSecret(f"https://{key_vault_name}.vault.azure.net/", "client_secret")
tenant_id = getSecret(f"https://{key_vault_name}.vault.azure.net/", "tenant_id")
authority = f"https://login.microsoftonline.com/{tenant_id}"
token_url = f"{authority}/oauth2/v2.0/token"

headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
}

data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
    'scope': 'https://graph.microsoft.com/.default'
}

response = requests.post(token_url, headers=headers, data=data)
access_token = response.json().get('access_token')

# Upload a file to OneDrive
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

file_path = 'abfss://f72f59d8-9015-4de8-bab1-573ef077c941@onelake.dfs.fabric.microsoft.com/acccfb51-5a60-4be1-b423-97f71aa4edec/Files/diabetesdataset/diabetes.csv'
with open(file_path, 'rb') as file:
    content = file.read()

upload_url = 'your one drive location'
response = requests.put(upload_url, headers=headers, data=content)

if response.status_code == 201:
    print("File uploaded successfully")
else:
    print(f"Error: {response.status_code}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
