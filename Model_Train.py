# %% [markdown]
# # Lumbar Spine Degenerative Condition Classification Modelling and  Training 
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

# %% [markdown]
# ### My other Notebooks
# - [RNSA | EDA & Dataset Creation I ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-i) 
# - [RNSA | EDA & Dataset Creation II ](https://www.kaggle.com/code/archie40004/rsna-eda-dataset-creation-ii)
# - [RNSA | EDA & Dataset Creation III ](https://www.kaggle.com/code/archie40004/rsna-eda-dataset-creation-iii) 
# - [RSNA 2024 | RSNA | PreP & Modelling, Training ](https://www.kaggle.com/code/archie40004/rsna-prep-modelling/)<- you're reading now
# - [RSNA 2024 | RSNA | Submission](https://www.kaggle.com/code/archie40004/rsna-2024-submission/)

# %% [markdown]
# # 📚 | Import Libraries

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T09:18:32.152425Z","iopub.execute_input":"2024-10-09T09:18:32.152820Z","iopub.status.idle":"2024-10-09T09:18:46.365356Z","shell.execute_reply.started":"2024-10-09T09:18:32.152791Z","shell.execute_reply":"2024-10-09T09:18:46.364172Z"}}
!pip install dask

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T09:18:46.367462Z","iopub.execute_input":"2024-10-09T09:18:46.367830Z","iopub.status.idle":"2024-10-09T09:19:04.670323Z","shell.execute_reply.started":"2024-10-09T09:18:46.367799Z","shell.execute_reply":"2024-10-09T09:19:04.669526Z"}}
# packages

# standard
import numpy as np
import pandas as pd

import os
import time

# plots
import matplotlib.pyplot as plt
from matplotlib import animation, rc

import seaborn as sns

import cv2

import random 
import torch 

import warnings # warning handling
warnings.filterwarnings('ignore')
import glob
import json
import collections
import os

import keras_nlp
import keras
import tensorflow as tf

import numpy as np 
import pandas as pd
from tqdm import tqdm
import json

import matplotlib.pyplot as plt
import matplotlib as mpl

# %% [markdown]
# # ♻️ | Reproducibility 
# Sets value for random seed to produce similar result in each run.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T09:19:04.671494Z","iopub.execute_input":"2024-10-09T09:19:04.672127Z","iopub.status.idle":"2024-10-09T09:19:04.681293Z","shell.execute_reply.started":"2024-10-09T09:19:04.672097Z","shell.execute_reply":"2024-10-09T09:19:04.680378Z"}}
def set_random_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  

set_random_seed(42)

# %% [markdown]
# # ⚙️ | Modelling Schema

# %% [markdown]
# 
# 
# <div align="center">
#     <img src="https://i.ibb.co/KyL4Ljp/Workflow-Meth.png
# ">
# </div>

# %% [markdown]
# We chose this comprehensive workflow for developing a robust model to classify degenerative spine conditions using MRI data because it ensures a systematic and thorough approach to model development. Starting with data import and exploratory data analysis (EDA), we gain critical insights into the dataset, enabling effective preprocessing and feature engineering. By splitting the data into training, validation, and test sets, we ensure proper evaluation of model performance. The workflow incorporates iterative steps of model training, initial evaluation, hyperparameter tuning, and cross-validation to refine the models systematically. Error analysis and data collection (e.g. Agumentation) loops allow continuous improvement, while the inclusion of advanced techniques like coordinate integration, varaitional autoencoders, attention mechanisms, transformers, multi-head network ensures that our models capture both spatial and sequential information effectively. This structured methodology not only enhances model accuracy and robustness but also facilitates efficient deployment and ongoing monitoring, ensuring long-term success in real-world applications.

# %% [code] {"execution":{"iopub.status.busy":"2024-10-09T09:19:04.683001Z","iopub.execute_input":"2024-10-09T09:19:04.683260Z","iopub.status.idle":"2024-10-09T09:19:04.710269Z","shell.execute_reply.started":"2024-10-09T09:19:04.683238Z","shell.execute_reply":"2024-10-09T09:19:04.709249Z"},"jupyter":{"source_hidden":true}}
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class SpineNetMultiView(nn.Module):
    def __init__(self, num_conditions, num_levels, num_classes, class_weights):
        super(SpineNetMultiView, self).__init__()
        
        # ResNet-50 backbones for each view
        self.resnet_axial = models.resnet50(pretrained=True)
        self.resnet_sagittal = models.resnet50(pretrained=True)
        self.resnet_coronal = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layers
        self.resnet_axial.fc = nn.Identity()
        self.resnet_sagittal.fc = nn.Identity()
        self.resnet_coronal.fc = nn.Identity()
        
        # Additional layers for coordinate integration
        self.fc_coord_axial = nn.Linear(2, 512)
        self.fc_coord_sagittal = nn.Linear(2, 512)
        self.fc_coord_coronal = nn.Linear(2, 512)
        
        # Convolutional layer to combine image features and coordinates
        self.conv = nn.Conv2d(2048 * 3 + 512 * 3, 512, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_conditions * num_levels * num_classes)
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    def forward(self, x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal):
        # Extract features from each view
        x_axial = self.resnet_axial(x_axial)
        x_sagittal = self.resnet_sagittal(x_sagittal)
        x_coronal = self.resnet_coronal(x_coronal)
        
        # Process coordinates
        coords_axial = self.fc_coord_axial(coords_axial).unsqueeze(-1).unsqueeze(-1)
        coords_sagittal = self.fc_coord_sagittal(coords_sagittal).unsqueeze(-1).unsqueeze(-1)
        coords_coronal = self.fc_coord_coronal(coords_coronal).unsqueeze(-1).unsqueeze(-1)
        
        # Concatenate features and coordinates
        x_axial = torch.cat([x_axial, coords_axial.expand(-1, -1, x_axial.size(2), x_axial.size(3))], dim=1)
        x_sagittal = torch.cat([x_sagittal, coords_sagittal.expand(-1, -1, x_sagittal.size(2), x_sagittal.size(3))], dim=1)
        x_coronal = torch.cat([x_coronal, coords_coronal.expand(-1, -1, x_coronal.size(2), x_coronal.size(3))], dim=1)
        
        # Combine features from all views
        x = torch.cat([x_axial, x_sagittal, x_coronal], dim=1)
        
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 5, 5, 3)  # Reshape to (batch_size, 25 conditions/levels, 3 classes)
        return x

    def weighted_categorical_crossentropy(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        weights = torch.sum(self.class_weights * y_true, dim=-1)
        unweighted_losses = -torch.sum(y_true * torch.log(y_pred), dim=-1)
        weighted_losses = unweighted_losses * weights
        return torch.mean(weighted_losses)

# Model definition
class_weights = [1.0, 2.0, 4.0]  # Example class weights
model = SpineNetMultiView(num_conditions=5, num_levels=5, num_classes=3, class_weights=class_weights).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (assuming train_loader and valid_loader are defined)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal, labels) in enumerate(train_loader):
        x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal, labels = x_axial.cuda(), coords_axial.cuda(), x_sagittal.cuda(), coords_sagittal.cuda(), x_coronal.cuda(), coords_coronal.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal)
        loss = model.weighted_categorical_crossentropy(labels, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        for i, (x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal, labels) in enumerate(valid_loader):
            x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal, labels = x_axial.cuda(), coords_axial.cuda(), x_sagittal.cuda(), coords_sagittal.cuda(), x_coronal.cuda(), coords_coronal.cuda(), labels.cuda()

            outputs = model(x_axial, coords_axial, x_sagittal, coords_sagittal, x_coronal, coords_coronal)
            loss = model.weighted_categorical_crossentropy(labels, outputs)
            validation_loss += loss.item()

        print(f'Epoch {epoch + 1}, Validation loss: {validation_loss / len(valid_loader):.3f}')

print('Finished Training')

'''

# %% [markdown]
# # 🛠️ | Pipeline Implementation

# %% [markdown]
# ***All pipelines have the same common aspect as the baseline pipeline, the differences are in the data preproeccessing and model architecture.***

# %% [markdown]
# ## Baseline Pipeline

# %% [markdown]
# 
# <div align="center">
#     <img src="https://i.ibb.co/LYvBT93/Main-baseline.png">
# </div>

# %% [markdown]
# The piepline shows the different proccesses, please note that Dataset Creation and getting the Output Datasets are done in 
# - [RNSA | EDA & Dataset Creation I ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-i)
# 
# - [RNSA | EDA & Dataset Creation II ](https://www.kaggle.com/code/archie40004/rnsa-eda-dataset-creation-ii)

# %% [markdown]
# ## 📊 | Spine Dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T09:19:04.712998Z","iopub.execute_input":"2024-10-09T09:19:04.713272Z","iopub.status.idle":"2024-10-09T09:21:26.443440Z","shell.execute_reply.started":"2024-10-09T09:19:04.713250Z","shell.execute_reply":"2024-10-09T09:21:26.442403Z"}}
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
zip_path = '/kaggle/input/rnsa-2024-eda-dataset-creation-i/_output_.zip'
extract_to = './extracted_files'

# Unzip the file
unzip_file(zip_path, extract_to)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T10:29:02.637567Z","iopub.execute_input":"2024-10-09T10:29:02.638507Z","iopub.status.idle":"2024-10-09T10:29:02.662044Z","shell.execute_reply.started":"2024-10-09T10:29:02.638458Z","shell.execute_reply":"2024-10-09T10:29:02.661073Z"}}

class SpineDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.conditions = [
             'left_neural_foraminal_narrowing',
            'left_subarticular_stenosis','right_neural_foraminal_narrowing',
            'right_subarticular_stenosis','spinal_canal_stenosis'
        ]
        self.levels = ['l1/l2', 'l2/l3', 'l3/l4', 'l4/l5', 'l5/s1']
        
        # Pre-process data
        self.images, self.coordinates, self.labels, self.study_ids, self.valid_indices = self.process_data()

    def process_data(self):
        tqdm.pandas(desc="Loading and Processing Images")
        results = self.df.progress_apply(self.process_image, axis=1)
        
        images, coordinates, labels, study_ids, valid_indices = [], [], [], [], []
        for idx, result in enumerate(results):
            if result is not None:
                if self.mode == 'train':
                    img, coord, lbl, study_id = result
                    images.append(img)
                    coordinates.append(coord)
                    labels.append(lbl)
                    study_ids.append(study_id)
                else:  # test mode
                    img, study_id = result
                    images.append(img)
                    study_ids.append(study_id)
                valid_indices.append(idx)

        images = np.stack(images)
        if self.mode == 'train':
            coordinates = np.array(coordinates)
            labels = np.stack(labels)
        else:
            coordinates = None
            labels = None

        return images, coordinates, labels, np.array(study_ids), valid_indices

    def process_image(self, row):
        image_path = os.path.join(self.image_dir, str(row['study_id']), str(row['series_id']), f'{row["instance_number"]}.png')
        try:
            image = Image.open(image_path).convert('L')
            image = image.resize((224, 224))
            image_np = np.array(image)
            
            if self.mode == 'train':
                coordinates = [int(row['x'] * (224 / image.size[1])), int(row['y'] * (224 / image.size[0]))]
                label = self.get_label(row)
                return image_np, coordinates, label, row['study_id']
            else:
                return image_np, row['study_id']
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            return None

    def get_label(self, row):
        label = np.zeros((5, 5, 3))
        for i, cond in enumerate(self.conditions):
            for j, level in enumerate(self.levels):
                col_name = f'{cond}_{level}'
                if col_name in row and not pd.isna(row[col_name]):
                    severity = row[col_name]
                    if severity == 0:
                        label[i, j, 0] = 1
                    elif severity == 1:
                        label[i, j, 1] = 1
                    elif severity == 2:
                        label[i, j, 2] = 1
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image.astype(np.uint8), 'L')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            coordinate = torch.tensor(self.coordinates[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, coordinate, label
        else:
            return image

    def get_df_index(self, idx):
        return self.valid_indices[idx]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T10:29:04.864547Z","iopub.execute_input":"2024-10-09T10:29:04.865687Z","iopub.status.idle":"2024-10-09T10:34:23.901168Z","shell.execute_reply.started":"2024-10-09T10:29:04.865653Z","shell.execute_reply":"2024-10-09T10:34:23.899933Z"}}

transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the datasets
df_train_long = pd.read_pickle('/kaggle/input/rsna-2024-eda-dataset-creation-ii/train_long.pkl')
df_test_long = pd.read_pickle('/kaggle/input/rsna-2024-eda-dataset-creation-ii/test_long.pkl')

# Initialize datasets
train_dataset = SpineDataset(df_train_long, '/kaggle/working/extracted_files/train_png_images', transform=transform, mode='train')
test_dataset = SpineDataset(df_test_long, '/kaggle/working/extracted_files/test_png_images', transform=transform, mode='test')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T11:08:13.660933Z","iopub.execute_input":"2024-10-09T11:08:13.661733Z","iopub.status.idle":"2024-10-09T11:08:13.677044Z","shell.execute_reply.started":"2024-10-09T11:08:13.661699Z","shell.execute_reply":"2024-10-09T11:08:13.675973Z"}}
# Split train dataset into train and validation based on study_id
splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_indices, val_indices = next(splitter.split(X=train_dataset.images, groups=train_dataset.study_ids))

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

# Initialize data loaders
train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Print dataset sizes
print(f"Train dataset size: {len(train_subset)}")
print(f"Validation dataset size: {len(val_subset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %% [markdown]
# 

# %% [markdown]
# 
# <div align="center">
#     <img src="https://i.ibb.co/HhF9vkR/Baseline-Model-Arch.png">
# </div>

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T10:34:23.945022Z","iopub.execute_input":"2024-10-09T10:34:23.945340Z","iopub.status.idle":"2024-10-09T10:34:23.951538Z","shell.execute_reply.started":"2024-10-09T10:34:23.945315Z","shell.execute_reply":"2024-10-09T10:34:23.950548Z"}}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from sklearn.metrics import classification_report


# If you're using any data augmentation or transforms
import torchvision.transforms as transforms

# For progress tracking during training
from tqdm import tqdm

# For plotting if needed
import matplotlib.pyplot as plt

# For saving and loading models
import os
import json

# If you're using wandb for experiment tracking
# import wandb

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T10:34:23.953787Z","iopub.execute_input":"2024-10-09T10:34:23.954168Z","iopub.status.idle":"2024-10-09T10:34:24.251735Z","shell.execute_reply.started":"2024-10-09T10:34:23.954133Z","shell.execute_reply":"2024-10-09T10:34:24.250741Z"}}
import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SpineNet(nn.Module):
    def __init__(self, num_conditions=5, num_levels=5, num_classes=3):
        super(SpineNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.fc_coord = nn.Linear(2, 224*224)
        
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_conditions * num_levels * num_classes)

        # Add BatchNorm layers
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, coords):
        coords = self.fc_coord(coords).view(coords.size(0), 1, 224, 224)
        x = torch.cat([x, coords], dim=1)
        
        x = self.resnet(x)
        x = self.conv(x)
        x = self.bn1(x)  # Add BatchNorm
        x = nn.functional.relu(x)  # Add ReLU activation
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)  # Add BatchNorm
        x = nn.functional.relu(x)  # Add ReLU activation
        x = self.fc2(x)
        x = x.view(-1, 5, 5, 3)
        
        # Remove softmax from here
        return x
    

def weighted_categorical_crossentropy(outputs, targets, weights):
        # Reshape outputs and targets to (batch_size * 25, 3)
        outputs = outputs.view(-1, 3)
        targets = targets.view(-1, 3)

        # Apply log_softmax to outputs
        log_probs = nn.functional.log_softmax(outputs, dim=1)

        # Calculate weighted negative log-likelihood loss
        loss = -(targets * log_probs * weights.unsqueeze(0)).sum(dim=1).mean()

        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpineNet().to(device)
print(f"Model is running on: {device}")

# %% [markdown]
# # 🚂 | Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T10:34:24.253009Z","iopub.execute_input":"2024-10-09T10:34:24.253342Z","iopub.status.idle":"2024-10-09T10:55:10.027213Z","shell.execute_reply.started":"2024-10-09T10:34:24.253313Z","shell.execute_reply":"2024-10-09T10:55:10.026172Z"}}
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def calculate_accuracy(outputs, labels):
    outputs = outputs.view(-1, 3)
    labels = labels.view(-1, 3)
    
    _, predicted = torch.max(outputs, 1)
    _, true_classes = torch.max(labels, 1)
    
    correct = (predicted == true_classes).sum().item()
    total = labels.size(0)
    return correct / total


def train_model(model, train_loader, test_loader, val_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Reduce learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Define class weights
    weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        for images, coordinates, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, coordinates, labels = images.to(device), coordinates.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, coordinates)
            loss = weighted_categorical_crossentropy(outputs, labels, weights)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for images, coordinates, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, coordinates, labels = images.to(device), coordinates.to(device), labels.to(device)
                
                outputs = model(images, coordinates)
                loss = weighted_categorical_crossentropy(outputs, labels, weights)
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    return model


trained_model = train_model(model, train_loader, test_loader,val_loader)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T10:55:10.029011Z","iopub.execute_input":"2024-10-09T10:55:10.029399Z","iopub.status.idle":"2024-10-09T10:55:14.952734Z","shell.execute_reply.started":"2024-10-09T10:55:10.029365Z","shell.execute_reply":"2024-10-09T10:55:14.951417Z"}}
from sklearn.metrics import classification_report
import torch
import numpy as np

def evaluate_model(model, data_loader):
    model.eval()
    y_pred = []
    y_true = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for images, coordinates, labels in data_loader:
            images, coordinates, labels = images.to(device), coordinates.to(device), labels.to(device)
            outputs = model(images, coordinates)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    
    # Reshape predictions and true labels
    y_pred = y_pred.reshape(-1, 3)
    y_true = y_true.reshape(-1, 3)
    
    # Convert to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Print overall classification report
    print("Overall Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=['Normal/Mild', 'Moderate', 'Severe'],
                                zero_division=0))
    
    # If you want separate reports for each condition/level:
    conditions = [
        'left_neural_foraminal_narrowing',
        'left_subarticular_stenosis',
        'right_neural_foraminal_narrowing',
        'right_subarticular_stenosis',
        'spinal_canal_stenosis'
    ]
    levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    
    for i, condition in enumerate(conditions):
        for j, level in enumerate(levels):
            idx = i * 5 + j
            start = idx * 3
            end = (idx + 1) * 3
            
            y_true_subset = y_true_classes[start:end]
            y_pred_subset = y_pred_classes[start:end]
            
            # Get unique classes in this subset
            unique_classes = np.unique(np.concatenate([y_true_subset, y_pred_subset]))
            class_names = ['Normal/Mild', 'Moderate', 'Severe']
            target_names = [class_names[i] for i in unique_classes]
            
            print(f"\nClassification Report for {condition} at {level}:")
            try:
                print(classification_report(y_true_subset, y_pred_subset, 
                                            target_names=target_names,
                                            zero_division=0))
            except ValueError as e:
                print(f"Error generating report: {e}")
                print(f"Unique true classes: {np.unique(y_true_subset)}")
                print(f"Unique predicted classes: {np.unique(y_pred_subset)}")

# Evaluate the model
print("Evaluation on Validation Set:")
evaluate_model(trained_model, val_loader)


# %% [markdown]
# # 📬 | Submission

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-09T11:12:25.545938Z","iopub.execute_input":"2024-10-09T11:12:25.546504Z","iopub.status.idle":"2024-10-09T11:12:28.406481Z","shell.execute_reply.started":"2024-10-09T11:12:25.546448Z","shell.execute_reply":"2024-10-09T11:12:28.405153Z"}}
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

def generate_submission(model, data_loader, df_test, output_file='submission.csv'):
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            coordinates = torch.zeros(images.size(0), 2).to(device)
            outputs = model(images, coordinates)
            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=-1)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    conditions = [
        'left_neural_foraminal_narrowing',
        'left_subarticular_stenosis',
        'right_neural_foraminal_narrowing',
        'right_subarticular_stenosis',
        'spinal_canal_stenosis'
    ]
    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    
    
    submission_rows = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        study_id = int(row['study_id'])
        instance_number = row['instance_number']
        condition = row['condition']
        level = row['level']
        
        # Find the index of the condition and level
        condition_idx = conditions.index(condition)
        level_idx = levels.index(level)
        
        # Calculate the index in the predictions array
        pred_idx = i // 25  # Because we have 25 rows per image
        
        if pred_idx >= len(predictions):
            print(f"Warning: Index {pred_idx} is out of bounds. Using default probabilities.")
            probs = np.array([1/3, 1/3, 1/3])  # Default to equal probabilities
        else:
            probs = predictions[pred_idx, condition_idx, level_idx]
        
        row_id = f"{study_id}_{condition}_{level}"
        submission_rows.append({
            'row_id': row_id,
            'normal_mild': probs[0],
            'moderate': probs[1],
            'severe': probs[2]
        })
    
    submission = pd.DataFrame(submission_rows)
    submission = submission.groupby('row_id').mean().reset_index()
    
    submission.to_csv(output_file, index=False)
    return submission

# Generate predictions for submission
sub = generate_submission(trained_model, test_loader, df_test_long)

# Print the first few rows of sub to verify its content
print(sub)

# Verify that probabilities sum to 1 (or very close to 1 due to floating-point precision)
print("\nSum of probabilities for first few rows:")
print(sub[['normal_mild', 'moderate', 'severe']].sum(axis=1).head(25))

# Save the file
output_path = '/kaggle/working/submission.csv'
sub.to_csv(output_path, index=False)

print(f"\nFile saved at: {output_path}")
print(f"Total rows in submission: {len(sub)}")
