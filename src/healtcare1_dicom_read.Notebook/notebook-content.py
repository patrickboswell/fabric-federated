# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "90dbc64c-fe0b-4545-aea6-70e71a87649b",
# META       "default_lakehouse_name": "healthcare1_msft_bronze",
# META       "default_lakehouse_workspace_id": "dd0170a3-64b2-476a-aa18-8d3209238968"
# META     }
# META   }
# META }

# CELL ********************

%pip install pydicom

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pydicom
import matplotlib.pyplot as plt
import zipfile
import os
from io import StringIO, BytesIO
import fsspec

from numpy import random
from matplotlib import animation
from IPython.display import HTML

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Welcome to your new notebook
# Type here in the cell editor to add code!

df = spark.sql("SELECT * FROM healthcare1_msft_bronze.dicomimagingmetastore where accessionnumber = 'a508258761846499' and seriesinstanceuid ='1.3.6.1.4.1.14519.5.2.1.6450.9002.323744164395727327590329158420' order by instancenumber asc")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def unzip(zip_archive):
    """
    zip_archive is a zipfile object (from
        zip_archive = zipfile.ZipFile(filename, 'r') for example)

    Returns a dictionary of file names and file like objects (BytesIO's)

    The filter in the if statement skips directories and dot files
    """

    file_list = []
    for file_name in zip_archive.namelist():
        if (not os.path.basename(file_name).startswith('.') and
                not file_name.endswith('/')):
            file_object = zip_archive.open(file_name)
            file_like_object = BytesIO(file_object.read())
            ds = pydicom.dcmread(file_like_object)
            file_object.close()
            file_like_object.seek(0)
            name = os.path.basename(file_name)
            file_list.append((name, file_like_object))
    return file_list, ds



def open_abfss(tenant):
    # Inputs
    filesystem_code = "abfss"           # Azure Blob File System Secure
    onelake_account_name = tenant    # This is "onelake" for everyone
    onelake_host = "onelake.dfs.fabric.microsoft.com"

    # Initiate filesystem
    onelake_filesystem_class = fsspec.get_filesystem_class(filesystem_code)

    onelake_filesystem = onelake_filesystem_class(
        account_name=onelake_account_name,
        account_host=onelake_host)

    return onelake_filesystem


def unzip(paths, abfss_reader):
    """
    zip_archive is a zipfile object (from
        zip_archive = zipfile.ZipFile(filename, 'r') for example)

    Returns a dictionary of file names and file like objects (BytesIO's)

    The filter in the if statement skips directories and dot files
    """

    file_list = []

    for path in paths:
        file = abfss_reader.open(path)
        zip_file = zipfile.ZipFile(file)
        for file_name in zip_file.namelist():
            if (not os.path.basename(file_name).startswith('.') and
                    not file_name.endswith('/')):
                file_object = zip_file.open(file_name)
                file_like_object = BytesIO(file_object.read())
                ds = pydicom.dcmread(file_like_object)
                arr = ds.pixel_array
                file_object.close()
                name = os.path.basename(file_name)
                file_list.append((name, ds, arr))
    return file_list

def plt_dcm(arr):
    plt.imshow(arr)

    plt.show() 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# CELL ********************

abfss_reader = open_abfss("dd0170a3-64b2-476a-aa18-8d3209238968")

paths = [str(row.filepath) for row in df.collect()]

images = unzip(paths, abfss_reader)

study_arr = [arr for name, img, arr in images]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

fig = plt.figure(f'{images[0][0]}')
frames = [[plt.imshow(img, animated=True)] for img in study_arr]

ani = animation.ArtistAnimation(fig, frames, blit=True)

HTML(ani.to_jshtml())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filelist, ds = unzip(zipfile.ZipFile('/lakehouse/default/Files/Process/Imaging/DICOM/2024/09/16/1726512548.679189_IM-0026-0043.dcm.zip'))

arr = ds.pixel_array

plt.imshow(arr)
plt.show() 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
