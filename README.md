# 🧠 RSNA 2024 Machine Learning Pipeline Overview

## 📊 Dataset Creation and Preprocessing

1. **Data Loading**: 
   - Train and test data loaded from pickle files (`train_long.pkl`, `test_long.pkl`)
   - Images accessed from PNG format (converted from original DICOM files)

2. **Custom Dataset Class (SpineDataset)**:
   - Handles image loading, resizing to 224x224 pixels
   - Coordinates scaling to match resized images
   - Label encoding for multi-class classification

3. **Data Splitting**:
   - Train-validation split using GroupShuffleSplit (90% train, 10% validation)
   - Grouping by study_id to prevent data leakage

4. **Data Loaders**:
   - Batch size: 64
   - Shuffling enabled for training data
   - 4 worker threads for parallel data loading

## 🏗️ Model Architecture (SpineNet)

- **Backbone**: ResNet18 (pretrained)
- **Input Modification**: 
  - Conv1 layer modified to accept 2 channels (image + coordinate channel)
- **Coordinate Integration**: 
  - FC layer to transform coordinates to image-like tensor
- **Custom Layers**:
  - Additional convolutional layer (512 -> 256 channels)
  - Global Average Pooling
  - Fully connected layers (256 -> 128 -> 75 outputs)
- **Batch Normalization**: Added after conv and first FC layer
- **Output**: 5 conditions x 5 levels x 3 classes = 75 total outputs

## 🚂 Training Process

- **Loss Function**: Weighted Categorical Cross-Entropy
  - Weights: [1.0, 2.0, 4.0] for [Normal/Mild, Moderate, Severe]
- **Optimizer**: Adam with initial learning rate 0.01
- **Learning Rate Scheduling**: ReduceLROnPlateau
  - Patience: 2 epochs
  - Factor: Not specified (default 0.1)
- **Gradient Clipping**: Max norm 1.0
- **Epochs**: 10
- **Best Learning Rate**: 0.000100 (final epoch)

## 📈 Evaluation Metrics

1. **Loss**: 
   - Train Loss: 1.0160
   - Validation Loss: 1.0673

2. **Accuracy**:
   - Train Accuracy: 0.7467 (74.67%)
   - Validation Accuracy: 0.7336 (73.36%)

3. **Overall Classification Report**:
   - Normal/Mild: Precision 0.86, Recall 0.86, F1-score 0.86
   - Moderate: Precision 0.34, Recall 0.23, F1-score 0.27
   - Severe: Precision 0.23, Recall 0.45, F1-score 0.31
   - Weighted Avg: Precision 0.74, Recall 0.73, F1-score 0.73

4. **Per-condition and Per-level Reports**:
   - Varied performance across different conditions and levels
   - Some categories show perfect classification (1.00 for all metrics)
   - Others show poor performance (0.00 for some metrics)

## 🔍 Analysis

1. **Class Imbalance**: 
   - Evident from the support numbers (Normal/Mild: 95983, Moderate: 19383, Severe: 7134)
   - Model performs best on the majority class (Normal/Mild)

2. **Overfitting**: 
   - Slight overfitting observed (Train Acc > Val Acc)
   - Could benefit from regularization techniques

3. **Performance Variability**: 
   - Significant variation in performance across different conditions and levels
   - Some categories have very few samples, leading to unreliable metrics

4. **Areas for Improvement**:
   - Address class imbalance (e.g., oversampling, class weights adjustment)
   - Increase model capacity or use transfer learning with a larger backbone
   - Implement data augmentation to improve generalization
   - Consider ensemble methods to boost performance

Overall, the model shows promising results with room for improvement, especially in handling class imbalance and rare categories.
