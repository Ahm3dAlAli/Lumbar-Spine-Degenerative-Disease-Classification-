# %% [markdown]
# # Lumbar Spine Degenerative Condition Classification Explantory Data Analysis and Data Creation I
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
# - [RNSA | EDA & Dataset Creation I ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-i) <- you're reading now
# - [RNSA | EDA & Dataset Creation II ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-ii) 
# - [RSNA 2024 | RSNA | PreP & Modelling, Training ](https://www.kaggle.com/code/archie40004/rsna-prep-modelling/) 

# %% [markdown]
# # 📚 | Import Libraries

# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T06:53:31.808886Z","iopub.execute_input":"2024-07-29T06:53:31.810410Z","iopub.status.idle":"2024-07-29T06:54:08.151141Z","shell.execute_reply.started":"2024-07-29T06:53:31.810367Z","shell.execute_reply":"2024-07-29T06:54:08.149688Z"}}
!pip install dask


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:08.153462Z","iopub.execute_input":"2024-07-29T06:54:08.153822Z","iopub.status.idle":"2024-07-29T06:54:10.531024Z","shell.execute_reply.started":"2024-07-29T06:54:08.153788Z","shell.execute_reply":"2024-07-29T06:54:10.529943Z"}}

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

# Plots
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import plotly.express as px
import seaborn as sns
import plotly.express as px
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

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import bz2
import pickle
import gc
import os
import re
import glob
from tqdm import tqdm

# %% [markdown]
# # ♻️ | Reproducibility 
# Sets value for random seed to produce similar result in each run.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.532460Z","iopub.execute_input":"2024-07-29T06:54:10.533011Z","iopub.status.idle":"2024-07-29T06:54:10.540482Z","shell.execute_reply.started":"2024-07-29T06:54:10.532980Z","shell.execute_reply":"2024-07-29T06:54:10.537975Z"}}
def set_random_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) 

set_random_seed(42)


# %% [markdown]
# # 📁 | Dataset Path

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.544018Z","iopub.execute_input":"2024-07-29T06:54:10.544505Z","iopub.status.idle":"2024-07-29T06:54:10.728378Z","shell.execute_reply.started":"2024-07-29T06:54:10.544457Z","shell.execute_reply":"2024-07-29T06:54:10.727162Z"}}
Path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'
train_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
test_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

df_train_main = pd.read_csv(Path+'/train.csv')
df_train_label = pd.read_csv(Path+'/train_label_coordinates.csv')
df_train_desc = pd.read_csv(Path+'/train_series_descriptions.csv')
df_test_desc = pd.read_csv(Path+'/test_series_descriptions.csv')


# %% [markdown]
# # 📖 | Meta Data
# 
# The dataset comprises MRI scans of the lumbar spine annotated by spine radiology specialists. The goal is to develop AI models to identify and classify degenerative conditions affecting the lumbar spine. The training dataset includes `2,000` MR studies with annotations, and the test set contains an undisclosed number of studies to be evaluated.
# 
# ## Files
# 
# ### `train.csv`
# - `study_id`: Unique identifier for each MR study. Each study may include multiple series of images.
# - `spinal_canal_stenosis_[l1_l2, l2_l3, l3_l4, l4_l5, l5_s1]`: Severity labels for spinal canal stenosis at each vertebral level.
# - `left_neural_foraminal_narrowing_[l1_l2, l2_l3, l3_l4, l4_l5, l5_s1]`: Severity labels for left neural foraminal narrowing at each vertebral level.
# - `right_neural_foraminal_narrowing_[l1_l2, l2_l3, l3_l4, l4_l5, l5_s1]`: Severity labels for right neural foraminal narrowing at each vertebral level.
# - `left_subarticular_stenosis_[l1_l2, l2_l3, l3_l4, l4_l5, l5_s1]`: Severity labels for left subarticular stenosis at each vertebral level.
# - `right_subarticular_stenosis_[l1_l2, l2_l3, l3_l4, l4_l5, l5_s1]`: Severity labels for right subarticular stenosis at each vertebral level.
# 
# ### `train_label_coordinates.csv`
# - `study_id`: Unique identifier for each MR study.
# - `series_id`: Identifier for the imagery series within each study.
# - `instance_number`: The image's order number within the 3D stack.
# - `condition`: The specific condition annotated (e.g., spinal canal stenosis, neural foraminal narrowing, subarticular stenosis).
# - `level`: The vertebral level related to the annotation (e.g., l3_l4).
# - `x`: The x-coordinate for the center of the labeled area.
# - `y`: The y-coordinate for the center of the labeled area.
# 
# ### `sample_submission.csv`
# - `row_id`: Unique identifier for each prediction, formatted as `study_id_condition_level` (e.g., `12345_spinal_canal_stenosis_l3_l4`).
# - `normal_mild`: Probability prediction for the condition being Normal/Mild.
# - `moderate`: Probability prediction for the condition being Moderate.
# - `severe`: Probability prediction for the condition being Severe.
# 
# ### `train/test_images/[study_id]/[series_id]/[instance_number].dcm`
# - Contains the MRI imagery data in DICOM format.
# 
# ### `train/test_series_descriptions.csv`
# - `study_id`: Unique identifier for each MR study.
# - `series_id`: Identifier for the imagery series within each study.
# - `series_description`: Description of the scan's orientation (e.g., Axial T2, Sagittal T1, Sagittal T2/STIR).
# 
# > Note that each study may contain multiple series and images, providing a comprehensive view of the lumbar spine. The annotations guide the identification of specific degenerative conditions and their severity at different vertebral levels.

# %% [markdown]
# The dataset includes MRI scans of the lumbar spine with annotations for various degenerative conditions. The key components of the dataset are:
# - **MRI Images* : Stored in DICOM format, organized by study ID, series ID, and instance number.
# - Annotations: Severity labels for conditions such as spinal canal stenosis, neural foraminal narrowing, and subarticular stenosis at different vertebral levels (L1/L2 to L5/S1).
# - Coordinates: X and Y coordinates for the center of the labeled areas in the images.
# - Severity Scores: Labels indicating the severity of conditions (Normal/Mild, Moderate, Severe).

# %% [markdown]
# # 🎨 | Exploratory Data Analysis (EDA)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.729701Z","iopub.execute_input":"2024-07-29T06:54:10.730053Z","iopub.status.idle":"2024-07-29T06:54:10.768876Z","shell.execute_reply.started":"2024-07-29T06:54:10.730021Z","shell.execute_reply":"2024-07-29T06:54:10.767695Z"}}
# structure of train
df_train_main.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.770528Z","iopub.execute_input":"2024-07-29T06:54:10.770896Z","iopub.status.idle":"2024-07-29T06:54:10.795189Z","shell.execute_reply.started":"2024-07-29T06:54:10.770866Z","shell.execute_reply":"2024-07-29T06:54:10.793961Z"}}
df_train_label.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.796747Z","iopub.execute_input":"2024-07-29T06:54:10.797134Z","iopub.status.idle":"2024-07-29T06:54:10.810178Z","shell.execute_reply.started":"2024-07-29T06:54:10.797092Z","shell.execute_reply":"2024-07-29T06:54:10.808812Z"}}
df_train_desc.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.811569Z","iopub.execute_input":"2024-07-29T06:54:10.811914Z","iopub.status.idle":"2024-07-29T06:54:10.848777Z","shell.execute_reply.started":"2024-07-29T06:54:10.811884Z","shell.execute_reply":"2024-07-29T06:54:10.847555Z"}}
# look at categories
for f in ['instance_number','condition','level']:
    print(df_train_label[f].value_counts())
    print('-'*50);print();

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:10.850305Z","iopub.execute_input":"2024-07-29T06:54:10.851290Z","iopub.status.idle":"2024-07-29T06:54:11.094082Z","shell.execute_reply.started":"2024-07-29T06:54:10.851243Z","shell.execute_reply":"2024-07-29T06:54:11.093000Z"}}
# join first two tables
df_train_step_1 = pd.merge(left=df_train_label, right=df_train_main, how='left', on='study_id').reset_index(drop=True)
# join with third table
df_train = pd.merge(left=df_train_step_1, right=df_train_desc, how='left', on=['study_id', 'series_id']).reset_index(drop=True)
df_train.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:11.098444Z","iopub.execute_input":"2024-07-29T06:54:11.098831Z","iopub.status.idle":"2024-07-29T06:54:11.112116Z","shell.execute_reply.started":"2024-07-29T06:54:11.098797Z","shell.execute_reply":"2024-07-29T06:54:11.110695Z"}}
# convert identifiers to categorical
df_train.study_id = df_train.study_id.astype('category')
df_train.series_id = df_train.series_id.astype('category')


df_test_desc.study_id = df_test_desc.study_id.astype('category')
df_test_desc.series_id = df_test_desc.series_id.astype('category')


# %% [markdown]
# ### Coordinates distributions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:11.113996Z","iopub.execute_input":"2024-07-29T06:54:11.114424Z","iopub.status.idle":"2024-07-29T06:54:13.195385Z","shell.execute_reply.started":"2024-07-29T06:54:11.114391Z","shell.execute_reply":"2024-07-29T06:54:13.193732Z"}}
# Create a scatter plot with Plotly
fig = px.scatter(df_train, x='x', y='y', opacity=0.6)

# Customize the layout
fig.update_layout(
    xaxis_title='X Coordinate',
    yaxis_title='Y Coordinate',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# Show the plot
fig.show()

# %% [markdown]
# ### Plot coordinate distributions with colored by condition

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:13.196935Z","iopub.execute_input":"2024-07-29T06:54:13.197330Z","iopub.status.idle":"2024-07-29T06:54:13.366435Z","shell.execute_reply.started":"2024-07-29T06:54:13.197288Z","shell.execute_reply":"2024-07-29T06:54:13.364968Z"}}

# Define custom color palette
custom_palette = ['#4600c0',  '#00a419', '#111111', '#ffff00', '#ff0000']

# Create the plot with Plotly
fig = px.scatter(df_train, x='x', y='y', color='condition', 
                 color_discrete_sequence=custom_palette, 
                 opacity=0.6, 
                 labels={'x': 'X Coordinate', 'y': 'Y Coordinate'})

# Update the layout for better readability
fig.update_layout(
    xaxis=dict(range=[0, df_train['x'].max()]),
    yaxis=dict(range=[0, df_train['y'].max()]),
    legend=dict(title='Condition', x=1.05, y=1),
    width=7*100, height= 7*100
)

# Show the plot
fig.show()

# %% [markdown]
# ### Plot coordinates distributions with colored by level

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:13.367889Z","iopub.execute_input":"2024-07-29T06:54:13.368280Z","iopub.status.idle":"2024-07-29T06:54:13.522520Z","shell.execute_reply.started":"2024-07-29T06:54:13.368241Z","shell.execute_reply":"2024-07-29T06:54:13.520961Z"}}

# Define custom color palette
custom_palette = ['#4600c0',  '#00a419', '#111111', '#ffff00', '#ff0000']

# Define plot size
figs_x = 5  
figs_y = 5

# Create the plot with Plotly
fig = px.scatter(df_train, x='x', y='y', color='level', 
                 color_discrete_sequence=custom_palette, 
                 opacity=0.4,
                 labels={'x': 'X Coordinate', 'y': 'Y Coordinate'})

# Update the layout for better readability
fig.update_layout(
    xaxis=dict(range=[0, df_train['x'].max()]),
    yaxis=dict(range=[0, df_train['y'].max()]),
    legend=dict(title='Level', x=1.05, y=1, traceorder='reversed'),
    width=figs_x*100, height=figs_y*100
)

# Show the plot
fig.show()

# %% [markdown]
# ### Plot stenosis distributions with colored by MR-split

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:13.524016Z","iopub.execute_input":"2024-07-29T06:54:13.524433Z","iopub.status.idle":"2024-07-29T06:54:13.654194Z","shell.execute_reply.started":"2024-07-29T06:54:13.524401Z","shell.execute_reply":"2024-07-29T06:54:13.652881Z"}}

# Define custom color palette
custom_palette = ['#000000', '#4600c0', '#ff0000']

# Define plot size
figs_x = 5  
figs_y = 5

# Create the plot with Plotly
fig = px.scatter(df_train, x='x', y='y', color='series_description', 
                 color_discrete_sequence=custom_palette, 
                 opacity=0.4,
                 labels={'x': 'X Coordinate', 'y': 'Y Coordinate'})

# Update the layout for better readability
fig.update_layout(
    xaxis=dict(range=[0, df_train['x'].max()]),
    yaxis=dict(range=[0, df_train['y'].max()]),
    legend=dict(title='Series Description', x=1.05, y=1),
    width=figs_x*100, height=figs_y*100
)

# Show the plot
fig.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:13.655793Z","iopub.execute_input":"2024-07-29T06:54:13.656149Z","iopub.status.idle":"2024-07-29T06:54:16.597013Z","shell.execute_reply.started":"2024-07-29T06:54:13.656119Z","shell.execute_reply":"2024-07-29T06:54:16.595733Z"}}
numerical_features =   df_train.series_description.unique()
gs=200
n=2 # num of columns
a=0 
k=1;
colorlabels = 'darkblue'

Label_size = 10 # Size font of xy labels
Title_size = 20 # Size font of Title
figs_x=13   
figs_y=5
plt.figure(figsize=(figs_x, figs_y))    
plt.suptitle("xy-Density of cases in plane-split", fontsize=Title_size+2, fontweight='bold', y=1.015)
for i in numerical_features:
        
        d = df_train[df_train.series_description == i]
        plt.subplot(1,n, k)
        #sns.jointplot(data=d, x='x', y='y', color='white', alpha=0.25)
        plt.hexbin(data=d, x='x', y='y',  gridsize=gs, cmap='CMRmap', bins='log', alpha = 1)
        
        #plt.colorbar(label='count in bin')
        plt.colorbar().set_label(label='count in bin',size=10, color  = 'grey')
        #plt.colorbar(size=8)
        
        plt.tick_params(axis='x', labelsize=Label_size)
        plt.tick_params(axis='y', labelsize=Label_size)
        
        plt.xlim([0, df_train.x.max()])
        plt.ylim([0, df_train.y.max()])
        plt.xlabel(f'x', fontsize=Label_size, color = colorlabels)
        plt.ylabel(f'y', fontsize=Label_size, color = colorlabels) 
        
        plt.title(f'{i}', color='black', fontsize=Title_size)
        k=k+1
        if k == (n+1):    
            k=1
            plt.show()
            plt.figure(figsize=(figs_x, figs_y))

# %% [markdown]
# ### Target distribution

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:16.598640Z","iopub.execute_input":"2024-07-29T06:54:16.598984Z","iopub.status.idle":"2024-07-29T06:54:32.429431Z","shell.execute_reply.started":"2024-07-29T06:54:16.598951Z","shell.execute_reply":"2024-07-29T06:54:32.427041Z"}}
train_label_df = df_train_label.copy()
train_data_df = df_train_main.copy()
train_label_df['new_col'] = df_train_label['condition'].str.lower().str.replace(' ', '_') +  '_' + df_train_label['level'].str.lower().str.replace('/', '_')

# Step 2: Merge the values from train_data_df based on study_id and the newly created column names
def get_target_value(row):
    study_id = row['study_id']
    new_col = row['new_col']
    return train_data_df[train_data_df['study_id'] == study_id][new_col].values[0]

train_label_df['target'] = train_label_df.apply(get_target_value, axis=1)

# Drop the 'new_col' column 
train_label_df.drop(columns=['new_col'], inplace=True)

# Copy the train_label_df to new DataFrame
final_train_df = train_label_df.copy()

# Drop the 'new_col' column 
train_label_df.drop(columns=['target'], inplace=True)

#final_train_df.to_csv('final_train.csv')
final_train_df.head()

# %% [markdown]
# ### Plotting the distribution of the target variable:

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.430696Z","iopub.status.idle":"2024-07-29T06:54:32.431143Z","shell.execute_reply.started":"2024-07-29T06:54:32.430935Z","shell.execute_reply":"2024-07-29T06:54:32.430953Z"}}
# Define custom color palette
p = ['#d0d0d0', '#ffba07', '#ff0000']

# Create the plot with Plotly
fig = px.histogram(final_train_df, x='target', color='target', 
                   color_discrete_sequence=p,
                   labels={'target': 'Severity'})

# Update the layout for better readability
fig.update_layout(
    xaxis_title='Severity Level',
    yaxis_title='Count',
    legend_title='Severity',
    title_font_size=15,
    title_font_family='Arial',
    title_font_color='blue',
    title_x=0.5,
    legend=dict(x=0.8, y=0.9, traceorder='normal', bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black')
)

# Show the plot
fig.show()

# %% [markdown]
# ### Plotting the distribution of the target variable in L/L splits:

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.432903Z","iopub.status.idle":"2024-07-29T06:54:32.433330Z","shell.execute_reply.started":"2024-07-29T06:54:32.433106Z","shell.execute_reply":"2024-07-29T06:54:32.433121Z"}}
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Define custom color palette
colours = {"Normal/Mild": "#273c75", "Moderate": "#44bd32", "Severe": "#ff0000"}

# Initialize subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=("Foraminal Distribution", "Subarticular Distribution", "Canal Distribution"))

# Define categories to plot
categories = ['foraminal', 'subarticular', 'canal']

# Iterate over categories and create bar plots
for idx, d in enumerate(categories):
    diagnosis = list(filter(lambda x: x.find(d) > -1, df_train_main.columns))
    dff = df_train_main[diagnosis]
    value_counts = dff.apply(pd.value_counts).fillna(0).T
    
    for severity in value_counts.columns:
        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts[severity],
            name=severity,
            marker_color=colours[severity],
            showlegend=(idx == 0)  # Only show legend for the first subplot
        ), row=1, col=idx + 1)

# Update layout
fig.update_layout(
    title_text="Severity Distribution by Condition",
    barmode='stack',
    legend_title_text='Severity',
    height=500,
    width=1200
)

# Show the plot
fig.show()

# %% [markdown]
# # 📷 | Images

# %% [markdown]
# A .dcm file follows the **Digital Imaging and Communications in Medicine** (DICOM) format. It is the standard format used for storing medical images and related metadata. It dates back to 1983, although it has been revised many times.
# 
# We can use the pydicom library to open and explore these files.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.434462Z","iopub.status.idle":"2024-07-29T06:54:32.434845Z","shell.execute_reply.started":"2024-07-29T06:54:32.434660Z","shell.execute_reply":"2024-07-29T06:54:32.434675Z"}}
train_label_coordinates=df_train_label

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.436313Z","iopub.status.idle":"2024-07-29T06:54:32.436855Z","shell.execute_reply.started":"2024-07-29T06:54:32.436581Z","shell.execute_reply":"2024-07-29T06:54:32.436604Z"}}
train_label_coordinates[train_label_coordinates.study_id==4003253]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.438453Z","iopub.status.idle":"2024-07-29T06:54:32.438980Z","shell.execute_reply.started":"2024-07-29T06:54:32.438705Z","shell.execute_reply":"2024-07-29T06:54:32.438727Z"}}
train_label_coordinates[train_label_coordinates.study_id==100206310]

# %% [markdown]
# ### Visualizing MR-images

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.440638Z","iopub.status.idle":"2024-07-29T06:54:32.441167Z","shell.execute_reply.started":"2024-07-29T06:54:32.440895Z","shell.execute_reply":"2024-07-29T06:54:32.440917Z"}}
train_label_coordinates['series_description'] = df_train.series_description

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.442772Z","iopub.status.idle":"2024-07-29T06:54:32.443327Z","shell.execute_reply.started":"2024-07-29T06:54:32.443026Z","shell.execute_reply":"2024-07-29T06:54:32.443049Z"}}
def mrt(id, ser, inst):
    lag=20
    path2 = Path +'/train_images/' + str(id) +'/' + str(ser)+'/' + str(inst) + '.dcm'

    ds = dicom.dcmread(path2)
    fig, ax = plt.subplots(figsize=(16, 8))
    from matplotlib.colors import LogNorm 

    ax.imshow(ds.pixel_array, cmap ='CMRmap')     # Display the image

    # Create a legend
    legend_elements = []

    # Plot the coordinates for the current condition
    ab = train_label_coordinates[(train_label_coordinates.study_id==id) & 
                                          (train_label_coordinates.instance_number==inst)&
                                         (train_label_coordinates.series_id==ser)]

    a = 25 * max(ds.pixel_array.shape)/640
    for _, row in ab.iterrows():
        x, y = row['x'], row['y']

        rect2 = patches.Rectangle((x - a, y - a), 2*a, 2*a, linewidth=2, edgecolor='white', facecolor='none')
        rect1 = patches.Rectangle((x - a, y - a), 2*a, 2*a, linewidth=2, facecolor='white', alpha = 0.25)

        ax.add_patch(rect2)
        ax.add_patch(rect1)

        # Add the condition to the legend
        legend_elements.append(patches.Patch(facecolor='none', edgecolor='r', ))

    # Add title
    title = f"{ab.series_description.unique()}, Study: {id}, Series: {ser}, Instance: {inst}"
    ax.set_title(title, fontsize=20)

    # Display additional columns:
    for _, row in ab.iterrows():
        text = f"level {row['level']}, {row['condition']}"
        ax.text(row['x'] + lag, row['y']+np.random.randint(-15, 15), text, fontsize=10, color='white', verticalalignment='center_baseline')
    
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.445563Z","iopub.status.idle":"2024-07-29T06:54:32.446131Z","shell.execute_reply.started":"2024-07-29T06:54:32.445815Z","shell.execute_reply":"2024-07-29T06:54:32.445860Z"}}
#Case #1
id = 100206310
ser= 1012284084
inst=8

mrt(id, ser, inst)

#Case #1
id = 100206310
ser= 1792451510
inst=8

mrt(id, ser, inst)

#Case #1
id = 100206310
ser= 2092806862
inst=8

mrt(id, ser, inst)

# %% [markdown]
# ### 3D MR-slides visualisation

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.447679Z","iopub.status.idle":"2024-07-29T06:54:32.448236Z","shell.execute_reply.started":"2024-07-29T06:54:32.447929Z","shell.execute_reply":"2024-07-29T06:54:32.447952Z"}}
def MR3d(id, ser):
    
    path_to_folder = Path + "/train_images/"+str(id)+'/'+str(ser)
    def load_dicom(path):
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data

    rc('animation', html='jshtml')

    def load_dicom(filename):
        ds = pydicom.dcmread(filename)
        return ds.pixel_array

    def load_dicom_line(path):
        t_paths = sorted(
            glob.glob(os.path.join(path, "*")), 
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]),
        )
        images = []
        for filename in t_paths:
            data = load_dicom(filename)
            if data.max() == 0:
                continue
            images.append(data)
        return images

    def create_animation(ims):
        fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(ims[0], cmap="CMRmap")
        text = plt.text(0.05, 0.05, f'Slide {1}', transform=fig.transFigure, fontsize=16, color='darkblue')

        def animate_func(i):
            im.set_array(ims[i])
            return [im]
        plt.title(f'id = {id}, series = {ser}')
        
        plt.close()  

        return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//10) #24

    images = load_dicom_line(path_to_folder)
    
    return create_animation(images)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-29T06:54:32.449551Z","iopub.status.idle":"2024-07-29T06:54:32.450060Z","shell.execute_reply.started":"2024-07-29T06:54:32.449796Z","shell.execute_reply":"2024-07-29T06:54:32.449818Z"}}
#Case #1
#id = 100206310
ser= 1012284084

#MR3d(id, ser)

# %% [markdown]
# # 📊 | Dataset Creation

# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T06:55:21.606514Z","iopub.execute_input":"2024-07-29T06:55:21.606949Z","iopub.status.idle":"2024-07-29T06:55:21.625483Z","shell.execute_reply.started":"2024-07-29T06:55:21.606916Z","shell.execute_reply":"2024-07-29T06:55:21.623872Z"}}
def reduce_mem_usage(df):
    """ Function to reduce the memory usage of a DataFrame by downcasting data types """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        if col == 'image':  # Skip the 'image' column
            continue

        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif pd.api.types.is_categorical_dtype(df[col]):
            if df[col].cat.ordered:
                df[col] = df[col].cat.as_ordered()
            else:
                df[col] = df[col].cat.as_unordered()

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df

# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T07:32:26.444870Z","iopub.execute_input":"2024-07-29T07:32:26.445385Z"}}
import os
import gc
import numpy as np
import glob
import pydicom
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from tqdm import tqdm
import pandas as pd

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def process_dicom_image(src_path, study_id, series_id, instance_number, output_dir):
    try:
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = Image.fromarray(image.astype(np.uint8)).convert('L')
        
        # Create output directory if it doesn't exist
        output_path = os.path.join(output_dir, str(study_id), str(series_id))
        os.makedirs(output_path, exist_ok=True)
        
        # Save the image as PNG
        image_file_path = os.path.join(output_path, f'{instance_number}.png')
        image.save(image_file_path)
    except Exception as e:
        print(f"Error processing file {src_path}: {e}")

def process_image_wrapper(args):
    process_dicom_image(*args)

def collect_tasks(df, image_dir, output_dir):
    tasks = []
    st_ids = df['study_id'].unique()
    for si in st_ids:
        pdf = df[df['study_id'] == si]
        for _, row in pdf.iterrows():
            series_id = row['series_id']
            img_paths = glob.glob(f'{image_dir}/{si}/{series_id}/*.dcm')
            img_paths = sorted(img_paths, key=natural_keys)
            for j, impath in enumerate(img_paths):
                instance_number = j + 1  # Assuming the instance number starts from 1
                tasks.append((impath, si, series_id, instance_number, output_dir))
    return tasks

def process_all_in_parallel(df, image_dir, output_dir):
    # Collect all tasks
    tasks = collect_tasks(df, image_dir, output_dir)
    
    # Process all tasks in parallel with progress bar
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_image_wrapper, task): task for task in tasks}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update(1)
    
    # Clear memory
    gc.collect()

# Clear memory before starting the process
gc.collect()

# Paths to data directories
train_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
test_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Output directories for PNG images
train_output_dir = 'train_png_images'
test_output_dir = 'test_png_images'

# Process all train tasks in parallel
process_all_in_parallel(df_train_desc, train_image_dir, train_output_dir)

# Process all test tasks in parallel
process_all_in_parallel(df_test_desc, test_image_dir, test_output_dir)


# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T06:55:24.609394Z","iopub.execute_input":"2024-07-29T06:55:24.609814Z","iopub.status.idle":"2024-07-29T07:04:35.268096Z","shell.execute_reply.started":"2024-07-29T06:55:24.609778Z","shell.execute_reply":"2024-07-29T07:04:35.266121Z"}}
'''
import os
import gc
import numpy as np
import glob
import pydicom
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import re
from tqdm import tqdm
import pandas as pd

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def process_dicom_image(src_path, study_id, series_id, instance_number, output_dir):
    try:
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = Image.fromarray(image.astype(np.uint8)).convert('L')
        
        # Create output directory if it doesn't exist
        output_path = os.path.join(output_dir, str(study_id), str(series_id))
        os.makedirs(output_path, exist_ok=True)
        
        # Save the image as PNG
        image_file_path = os.path.join(output_path, f'{instance_number}.png')
        image.save(image_file_path)

        return {'study_id': study_id, 'series_id': series_id, 'instance_number': instance_number, 'image_path': image_file_path}
    except Exception as e:
        print(f"Error processing file {src_path}: {e}")
        return None

def process_image_wrapper(args):
    return process_dicom_image(*args)

def collect_tasks(df, data_type='train'):
    tasks = []
    st_ids = df['study_id'].unique()
    for idx, si in enumerate(tqdm(st_ids, total=len(st_ids))):
        pdf = df[df['study_id'] == si]
        for _, row in pdf.iterrows():
            ds = row['series_description'].replace('/', '_')
            series_id = row['series_id']
            img_paths = glob.glob(f'/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/{data_type}_images/{si}/{series_id}/*.dcm')
            img_paths = sorted(img_paths, key=natural_keys)
            for j, impath in enumerate(img_paths):
                instance_number = j + 1  # Assuming the instance number starts from 1
                tasks.append((impath, si, series_id, instance_number, f'{data_type}_png_images'))
    return tasks

def process_and_save(tasks, chunk_size=50000):
    total_chunks = len(tasks) // chunk_size + 1
    for i in range(total_chunks):
        chunk_tasks = tasks[i*chunk_size:(i+1)*chunk_size]
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_image_wrapper, chunk_tasks), total=len(chunk_tasks)))
        gc.collect()

# Clear memory before starting the process
gc.collect()

# Paths to data directories
train_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
test_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Collect tasks for train and test data
train_tasks = collect_tasks(df_train_desc, data_type='train')
test_tasks = collect_tasks(df_test_desc, data_type='test')

# Process and save the train tasks
process_and_save(train_tasks)

# Process and save the test tasks
process_and_save(test_tasks)
'''

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-07-29T06:54:32.457331Z","iopub.status.idle":"2024-07-29T06:54:32.457732Z","shell.execute_reply.started":"2024-07-29T06:54:32.457535Z","shell.execute_reply":"2024-07-29T06:54:32.457551Z"}}
'''
import pandas as pd
import os
import gc
import numpy as np
import glob
import pydicom
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import re
import gzip
from tqdm import tqdm
from io import BytesIO
import pickle

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def process_dicom_image(src_path, study_id, series_id, instance_number):
    try:
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = Image.fromarray(image.astype(np.uint8)).convert('L')
        image_array = np.array(image, dtype=np.uint8)
        return {'study_id': study_id, 'series_id': series_id, 'instance_number': instance_number, 'image': image_array}
    except Exception as e:
        print(f"Error processing file {src_path}: {e}")
        return None

def process_image_wrapper(args):
    return process_dicom_image(*args)

def collect_tasks(df, data_type='train'):
    tasks = []
    st_ids = df['study_id'].unique()
    for idx, si in enumerate(tqdm(st_ids, total=len(st_ids))):
        pdf = df[df['study_id'] == si]
        for _, row in pdf.iterrows():
            ds = row['series_description'].replace('/', '_')
            series_id = row['series_id']
            img_paths = glob.glob(f'/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/{data_type}_images/{si}/{series_id}/*.dcm')
            img_paths = sorted(img_paths, key=natural_keys)
            for j, impath in enumerate(img_paths):
                instance_number = j + 1  # Assuming the instance number starts from 1
                tasks.append((impath, si, series_id, instance_number))
    return tasks


def process_and_save(tasks, output_file, chunk_size=50000):
    total_chunks = len(tasks) // chunk_size + 1
    for i in range(total_chunks):
        chunk_tasks = tasks[i*chunk_size:(i+1)*chunk_size]
        results = []
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image_wrapper, chunk_tasks), total=len(chunk_tasks)))
        # Filter out None results
        results = [res for res in results if res is not None]
        # Convert results to DataFrame
        chunk_df = pd.DataFrame(results)
        # Reduce memory usage
        chunk_df = reduce_mem_usage(chunk_df)
        
        # Save the chunk results as compressed pickle in memory
        compressed_file_path = f'{output_file}_part_{i}.pkl.gz'
        with gzip.open(compressed_file_path, 'wb') as f:
            pickle.dump(chunk_df, f)
        
        print(f'Saved {compressed_file_path}')
        del chunk_tasks
        del results
        del chunk_df
        gc.collect()

# Clear memory before starting the process
gc.collect()

# Paths to data directories
train_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
test_image_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Collect tasks for train and test data
train_tasks = collect_tasks(df_train_desc, data_type='train')
test_tasks = collect_tasks(df_test_desc, data_type='test')

# Process and save the train tasks in chunks
process_and_save(train_tasks, 'train_images')

# Process and save the test tasks in chunks
process_and_save(test_tasks, 'test_images')
'''


# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T06:54:32.459038Z","iopub.status.idle":"2024-07-29T06:54:32.459479Z","shell.execute_reply.started":"2024-07-29T06:54:32.459278Z","shell.execute_reply":"2024-07-29T06:54:32.459296Z"}}

# Define conditions and levels
conditions = [
    'spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 
    'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 
    'right_subarticular_stenosis'
]
levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

for condition in conditions:
    for level in levels:
        col_name = f'{condition}_{level}'
        if col_name not in df_train.columns:
            df_train[col_name] = np.nan  # or another appropriate default value

# Create long format DataFrame
df_long_train = pd.melt(df_train, 
                  id_vars=['study_id', 'series_id','series_description', 'instance_number'], 
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



# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T06:54:32.460835Z","iopub.status.idle":"2024-07-29T06:54:32.461226Z","shell.execute_reply.started":"2024-07-29T06:54:32.461022Z","shell.execute_reply":"2024-07-29T06:54:32.461038Z"}}
# Optimize memory usage of the long format DataFrames
df_long_train = reduce_mem_usage(df_long_train)
df_long_test = reduce_mem_usage(df_long_test)

# Save final DataFrames
df_long_train.to_pickle('train_long.pkl')
df_long_test.to_pickle('test_long.pkl')

# %% [code] {"execution":{"iopub.status.busy":"2024-07-29T06:54:32.462268Z","iopub.status.idle":"2024-07-29T06:54:32.462662Z","shell.execute_reply.started":"2024-07-29T06:54:32.462472Z","shell.execute_reply":"2024-07-29T06:54:32.462488Z"}}
df_long_test

# %% [markdown]
# Refrecnes
# 
# https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage
