# %% [markdown]
# # Lumbar Spine Degenerative Condition Classification Submission
# 
# <div align="center">
#     <img src="https://i.ibb.co/WKDHCCC/RSNA.png">
# </div>
# 
# In this competition, we aim to develop AI models that can accurately classify degenerative conditions of the lumbar spine using MRI images. Specifically, the objective is to create models that can simulate a radiologist's performance in diagnosing five key lumbar spine degenerative conditions: Left Neural Foraminal Narrowing, Right Neural Foraminal Narrowing, Left Subarticular Stenosis, Right Subarticular Stenosis, and Spinal Canal Stenosis.The goal of this project is to develop AI models to identify and classify degenerative conditions affecting the lumbar spine using MRI scans annotated by spine radiology specialists. Here’s a structured approach to tackle this project:
# 
# This guide will walk you through the process of building and fine-tuning models to detect and classify these conditions. Leveraging advanced machine learning frameworks and techniques, participants can create robust models capable of interpreting MRI scans with high accuracy.
# 
# **Did you know:**: This notebook is backend-agnostic? Which means it supports TensorFlow, PyTorch, and JAX backends. However, the best performance can be achieved with `JAX`. KerasNLP and Keras enable the choice of preferred backend. Explore further details on [Keras](https://keras.io/keras_3/).
# 
# **Note**: For a deeper understanding of KerasNLP, refer to the [KerasNLP guides](https://keras.io/keras_nlp/).
# 
# By participating in this challenge, you will contribute to the advancement of medical imaging and diagnostic radiology, potentially impacting patient care and treatment outcomes positively. Let's get started on building powerful AI models to enhance the detection and classification of lumbar spine degenerative conditions.
# ​

# %% [markdown]
# ### My other Notebooks
# - [RNSA 2024 | EDA & Dataset Creation I ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-i) 
# - [RNSA 2024 | EDA & Dataset Creation II ](https://www.kaggle.com/code/archie40004/rsna-eda-dataset-creation-ii)
# - [RSNA 2024 | RSNA | PreP & Modelling, Training](https://www.kaggle.com/code/archie40004/rsna-prep-modelling/)
# - [RSNA 2024 | RSNA | Submission](https://www.kaggle.com/code/archie40004/rsna-2024-submission/) <- you're reading now
# ​

# %% [code] {"execution":{"iopub.status.busy":"2024-09-18T09:51:29.206633Z","iopub.execute_input":"2024-09-18T09:51:29.207531Z","iopub.status.idle":"2024-09-18T09:54:05.846300Z","shell.execute_reply.started":"2024-09-18T09:51:29.207493Z","shell.execute_reply":"2024-09-18T09:54:05.845193Z"}}
'''
import zipfile
import os

def unzip_file(zip_path, extract_to):
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f"File {zip_path} does not exist")
        return
    
    # Create the directory to extract to if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Unzipping the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted all files to {extract_to}")

# Specify the path to the zip file and the directory to extract to
zip_path = '/kaggle/input/rsna-2024-prep-modelling-training/_output_.zip'
extract_to = './extracted_files'

# Unzip the file
unzip_file(zip_path, extract_to)
'''

# %% [markdown]
# # 📬 | Submission
# ​

# %% [code] {"execution":{"iopub.status.busy":"2024-10-09T11:42:49.158138Z","iopub.execute_input":"2024-10-09T11:42:49.158574Z","iopub.status.idle":"2024-10-09T11:42:49.652045Z","shell.execute_reply.started":"2024-10-09T11:42:49.158541Z","shell.execute_reply":"2024-10-09T11:42:49.650987Z"}}
import pandas as pd
import os
import shutil

# Define the input and output paths
input_path = '/kaggle/input/finalsubs/submission (8).csv'
output_path = '/kaggle/working/'

# Read the submission file
df = pd.read_csv(input_path)

# Rearrange the DataFrame to have row_id as the first column
cols = ['row_id'] + [col for col in df.columns if col != 'row_id']
df = df[cols]

# Function to remove directory contents
def clean_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Clean the output directory
clean_directory(output_path)

# Write the new submission file
df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)

print("Submission file has been processed and saved. All other files and folders in the output directory have been removed.")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-09T11:42:52.481528Z","iopub.execute_input":"2024-10-09T11:42:52.482023Z","iopub.status.idle":"2024-10-09T11:42:52.504017Z","shell.execute_reply.started":"2024-10-09T11:42:52.481976Z","shell.execute_reply":"2024-10-09T11:42:52.502756Z"}}
df.info()
