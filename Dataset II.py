# %% [markdown]
# # Lumbar Spine Degenerative Condition Classification Data Creation II
# 
# <div align="center">
#     <img src="https://i.ibb.co/WKDHCCC/RSNA.png">
# </div>
# 
# In this competition, we aim to develop AI models that can accurately classify degenerative conditions of the lumbar spine using MRI images. Specifically, the objective is to create models that can simulate a radiologist's performance in diagnosing five key lumbar spine degenerative conditions: Left Neural Foraminal Narrowing, Right Neural Foraminal Narrowing, Left Subarticular Stenosis, Right Subarticular Stenosis, and Spinal Canal Stenosis.The goal of this project is to develop AI models to identify and classify degenerative conditions affecting the lumbar spine using MRI scans annotated by spine radiology specialists. Here’s a structured approach to tackle this project:
# 
# This guide will walk you through the data and EDA neccessary to know how to handle the data.
# 
# **Did you know:**: This notebook is backend-agnostic? Which means it supports TensorFlow, PyTorch, and JAX backends. However, the best performance can be achieved with `JAX`. Explore further details on [Keras](https://keras.io/keras_3/).
# 
# 
# 
# By participating in this challenge, we will contribute to the advancement of medical imaging and diagnostic radiology, potentially impacting patient care and treatment outcomes positively. Let's get started on building powerful AI models to enhance the detection and classification of lumbar spine degenerative conditions.
# 
# 
# 
# 
# 
# In the this notebook we export datasets used.
# 
# 
# ### My other Notebooks
# - [RNSA | EDA & Dataset Creation I ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-i) 
# - [RNSA | EDA & Dataset Creation II ](https://www.kaggle.com/code/archie40004/rsna-eda-dataset-creation-ii) <- you're reading now
# - [RSNA 2024 | RSNA | PreP & Modelling, Training ](https://www.kaggle.com/code/archie40004/rsna-prep-modelling/)

# %% [markdown]
# # 📚 | Import Libraries

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T11:22:17.684816Z","iopub.execute_input":"2024-08-02T11:22:17.685923Z","iopub.status.idle":"2024-08-02T11:22:52.043522Z","shell.execute_reply.started":"2024-08-02T11:22:17.685856Z","shell.execute_reply":"2024-08-02T11:22:52.042207Z"},"jupyter":{"outputs_hidden":true}}
!pip install dask
!pip install pydicom
!pip install cupy

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-02T11:22:52.046474Z","iopub.execute_input":"2024-08-02T11:22:52.046917Z","iopub.status.idle":"2024-08-02T11:22:52.059550Z","shell.execute_reply.started":"2024-08-02T11:22:52.046868Z","shell.execute_reply":"2024-08-02T11:22:52.056407Z"}}

# Standard
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings # warning handling
warnings.filterwarnings('ignore')
import glob
import time
import collections
import os
import random

import cv2

# Dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import glob
from concurrent.futures import ProcessPoolExecutor

import re
import pydicom
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
import glob
from concurrent.futures import ProcessPoolExecutor
import pickle
import pydicom
import numpy as np
import cv2
import zlib
import pickle
import pandas as pd
import os
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import bz2
import pydicom as dicom
import matplotlib.patches as patches

# %% [markdown]
# # ♻️ | Reproducibility 
# Sets value for random seed to produce similar result in each run.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-02T11:22:52.061509Z","iopub.execute_input":"2024-08-02T11:22:52.062686Z","iopub.status.idle":"2024-08-02T11:22:52.080230Z","shell.execute_reply.started":"2024-08-02T11:22:52.062631Z","shell.execute_reply":"2024-08-02T11:22:52.078951Z"}}
def set_random_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) 

set_random_seed(42)

# %% [markdown]
# # 📁 | Dataset Path

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-02T11:22:52.082011Z","iopub.execute_input":"2024-08-02T11:22:52.082508Z","iopub.status.idle":"2024-08-02T11:22:52.177310Z","shell.execute_reply.started":"2024-08-02T11:22:52.082466Z","shell.execute_reply":"2024-08-02T11:22:52.176200Z"}}
Path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
train_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
test_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

df_train_main = pd.read_csv(Path+'/train.csv')
df_train_label = pd.read_csv(Path+'/train_label_coordinates.csv')
df_train_desc = pd.read_csv(Path+'/train_series_descriptions.csv')
df_test_desc = pd.read_csv(Path+'/test_series_descriptions.csv')

# %% [markdown]
# # 📊 | Dataset Creation

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T11:22:52.180254Z","iopub.execute_input":"2024-08-02T11:22:52.180637Z","iopub.status.idle":"2024-08-02T11:23:28.259733Z","shell.execute_reply.started":"2024-08-02T11:22:52.180604Z","shell.execute_reply":"2024-08-02T11:23:28.258569Z"}}

# Define a mapping function for levels and conditions
def get_relevant_column(row):
    condition = row['condition'].lower().replace(' ', '_')
    level = row['level'].replace('/', '_').lower()
    return f"{condition}_{level}"

# Add relevant column to each row
df_train_label['relevant_column'] = df_train_label.apply(get_relevant_column, axis=1)

# Prepare a function to create columns with NaN except for the relevant one
def assign_values(row, df_main):
    row_data = {col: np.nan for col in df_main.columns if col != 'study_id'}
    if row['relevant_column'] in df_main.columns:
        row_data[row['relevant_column']] = df_main.loc[df_main['study_id'] == row['study_id'], row['relevant_column']].values[0]
    return pd.Series(row_data)

# Create the filtered DataFrame
filtered_df = df_train_label.apply(lambda row: assign_values(row, df_train_main), axis=1)

# Merge with the label DataFrame
df_train_step_1 = pd.concat([df_train_label, filtered_df], axis=1)

# Drop the helper column
df_train_step_1.drop(columns=['relevant_column'], inplace=True)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-02T11:23:28.261034Z","iopub.execute_input":"2024-08-02T11:23:28.261357Z","iopub.status.idle":"2024-08-02T11:23:28.377326Z","shell.execute_reply.started":"2024-08-02T11:23:28.261329Z","shell.execute_reply":"2024-08-02T11:23:28.376260Z"}}

# join with third table
df_train = pd.merge(left=df_train_step_1, right=df_train_desc, how='left', on=['study_id', 'series_id']).reset_index(drop=True)

# convert identifiers to categorical
df_train.study_id = df_train.study_id.astype('category')
df_train.series_id = df_train.series_id.astype('category')


df_test_desc.study_id = df_test_desc.study_id.astype('category')
df_test_desc.series_id = df_test_desc.series_id.astype('category')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-02T11:23:28.454570Z","iopub.execute_input":"2024-08-02T11:23:28.455028Z","iopub.status.idle":"2024-08-02T11:23:28.470579Z","shell.execute_reply.started":"2024-08-02T11:23:28.454985Z","shell.execute_reply":"2024-08-02T11:23:28.469225Z"}}
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T11:37:41.548488Z","iopub.execute_input":"2024-08-02T11:37:41.548863Z","iopub.status.idle":"2024-08-02T11:37:44.144042Z","shell.execute_reply.started":"2024-08-02T11:37:41.548835Z","shell.execute_reply":"2024-08-02T11:37:44.142360Z"},"jupyter":{"outputs_hidden":true}}

# Define conditions and levels
conditions = [
    'spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 
    'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 
    'right_subarticular_stenosis'
]
levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']


# Create long format DataFrame
df_long_train = pd.melt(df_train, 
                  id_vars=['study_id', 'series_id','series_description', 'instance_number','x','y'], 
                  value_vars=[f'{cond}_{level}' for cond in conditions for level in levels], 
                  var_name='condition_level', 
                  value_name='target')


# Split the 'condition_level' column into 'condition' and 'level'
df_long_train['condition'] = df_long_train['condition_level'].apply(lambda x: '_'.join(x.split('_')[:-2]))
df_long_train['level'] = df_long_train['condition_level'].apply(lambda x: '_'.join(x.split('_')[-2:]).replace('_', '/'))
df_long_train = df_long_train.drop(columns=['condition_level'])


# Define the label map
label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df_long_train['target'] = df_long_train['target'].map(label_map)

df_long_train = df_long_train.dropna(subset=['target'])

df_long_train = df_long_train.sort_values(by=['study_id','series_id'])


# Expand the test DataFrame to have one row per image instance
test_data = []
for _, row in df_test_desc.iterrows():
    study_id = row['study_id']
    series_id = row['series_id']
    series_description = row['series_description']
    series_path = f'/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/{study_id}/{series_id}/'
    instance_files = sorted(glob.glob(f'{series_path}/*.dcm'), key=natural_keys)
    for instance_number, _ in enumerate(instance_files, start=1):
        for condition in conditions:
            for level in levels:
                test_data.append([study_id, series_id, series_description, instance_number, condition, level, 0])

df_long_test = pd.DataFrame(test_data, columns=['study_id', 'series_id', 'series_description', 'instance_number', 'condition', 'level', 'target'])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-02T11:23:28.636743Z","iopub.status.idle":"2024-08-02T11:23:28.637180Z","shell.execute_reply.started":"2024-08-02T11:23:28.636980Z","shell.execute_reply":"2024-08-02T11:23:28.636999Z"}}

# Save final DataFrames
df_long_train.to_pickle('train_long.pkl')
df_long_test.to_pickle('test_long.pkl')
