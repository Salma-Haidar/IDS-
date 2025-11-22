# NSL-KDD CNN-LSTM Intrusion Detection System - Kaggle Dataset Guide

## Overview
This guide will help you implement a high-performance CNN-LSTM intrusion detection system using the NSL-KDD dataset from Kaggle. The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset, specifically designed for network intrusion detection research.

## NSL-KDD Dataset Information

### About NSL-KDD
- **Purpose**: Network intrusion detection
- **Records**: ~148,000 training samples, ~22,000 test samples
- **Features**: 41 features + 1 label
- **Attack Types**: 4 main categories (DoS, Probe, R2L, U2R)
- **Advantages**: Balanced dataset, no redundant records

### Dataset Features (41 total)

#### Basic Features (9)
1. `duration` - Connection duration
2. `protocol_type` - Protocol used (TCP, UDP, ICMP)
3. `service` - Network service (HTTP, FTP, etc.)
4. `flag` - Connection status
5. `src_bytes` - Bytes sent from source
6. `dst_bytes` - Bytes sent to destination
7. `land` - Same host/port for source and destination
8. `wrong_fragment` - Wrong fragments
9. `urgent` - Urgent packets

#### Content Features (13)
10. `hot` - Hot indicators
11. `num_failed_logins` - Failed login attempts
12. `logged_in` - Successfully logged in
13. `num_compromised` - Compromised conditions
14. `root_shell` - Root shell obtained
15. `su_attempted` - Su root attempted
16. `num_root` - Root accesses
17. `num_file_creations` - File creation operations
18. `num_shells` - Shell prompts
19. `num_access_files` - Access control files
20. `num_outbound_cmds` - Outbound commands
21. `is_host_login` - Host login
22. `is_guest_login` - Guest login

#### Traffic Features (19)
23-41. Various connection and host-based statistical features

### Attack Categories
- **Normal**: Legitimate traffic
- **DoS**: Denial of Service attacks
- **Probe**: Surveillance and probing attacks  
- **R2L**: Remote to Local attacks
- **U2R**: User to Root attacks

## Step-by-Step Implementation Guide

### Step 1: Download NSL-KDD from Kaggle

1. **Visit Kaggle**: https://www.kaggle.com/datasets/hassan06/nslkdd
2. **Download**: Click "Download" to get the dataset
3. **Extract**: You'll get files like:
   - `KDDTrain+.txt` (Training data)
   - `KDDTest+.txt` (Test data)

### Step 2: Setup Your Environment

```bash
# Create project directory
mkdir nsl_kdd_intrusion_detection
cd nsl_kdd_intrusion_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install tensorflow==2.13.0 pandas numpy scikit-learn matplotlib seaborn
```

### Step 3: Project Structure

```
nsl_kdd_intrusion_detection/
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── processed/
├── models/
│   └── saved_models/
├── results/
│   ├── plots/
│   └── reports/
├── nsl_kdd_cnn_lstm_detector.py
└── requirements.txt
```

### Step 4: Data Understanding and Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data for exploration
train_data = pd.read_csv('data/KDDTrain+.txt', header=None)
print(f"Training data shape: {train_data.shape}")
print(f"Data types:\n{train_data.dtypes}")

# Check class distribution
print(f"Class distribution:\n{train_data.iloc[:, -1].value_counts()}")
```

### Step 5: Run the CNN-LSTM Model

```python
# Update file paths in the main script
train_path = "data/KDDTrain+.txt"
test_path = "data/KDDTest+.txt"

# Run the model
python nsl_kdd_cnn_lstm_detector.py
```

## Model Architecture Explanation

### CNN Block (Feature Extraction)
```python
# First CNN Layer
Conv1D(64 filters, kernel_size=3) → Extract local patterns
BatchNormalization() → Normalize features
MaxPooling1D(pool_size=2) → Reduce dimensionality
Dropout(0.3) → Prevent overfitting

# Second CNN Layer
Conv1D(128 filters, kernel_size=3) → Extract higher-level features
BatchNormalization() → Stabilize training
MaxPooling1D(pool_size=2) → Further dimensionality reduction
Dropout(0.3) → Regularization
```

### LSTM Block (Temporal Patterns)
```python
# Bi-directional LSTM
LSTM(100 units, return_sequences=True) → Capture long-term dependencies
LSTM(50 units) → Final temporal encoding
Dropout(0.2) → Prevent overfitting
```

### Dense Block (Classification)
```python
Dense(64, activation='relu') → Feature combination
BatchNormalization() → Stabilize learning
Dropout(0.4) → Regularization
Dense(32, activation='relu') → Final feature processing
Dense(1 or n_classes) → Classification output
```

## Expected Performance Results

### Binary Classification (Normal vs Attack)
- **Accuracy**: 95-97%
- **Precision**: 94-96%
- **Recall**: 93-95%
- **F1-Score**: 94-96%
- **AUC-ROC**: 0.96-0.98

### Multi-class Classification (Specific Attack Types)
- **Accuracy**: 92-94%
- **Precision**: 90-93%
- **Recall**: 89-92%
- **F1-Score**: 91-93%

## Customization Options

### 1. Hyperparameter Tuning
```python
# Model architecture
sequence_length = 10  # Try 5, 10, 15, 20
cnn_filters = [64, 128]  # Try [32, 64], [128, 256]
lstm_units = [100, 50]  # Try [50, 25], [200, 100]

# Training parameters
learning_rate = 0.001  # Try 0.0001, 0.01
batch_size = 64  # Try 32, 128, 256
epochs = 100  # Adjust based on convergence
```

### 2. Feature Selection Options
```python
# RFE parameters
use_rfe = True
n_features = 20  # Try 15, 25, 30
step = 1  # Features to remove each iteration
```

### 3. Classification Types
```python
# Binary classification (Normal vs Attack)
classification_type = 'binary'

# Multi-class classification (All attack types)
classification_type = 'multiclass'
```

## Advanced Features

### 1. Data Augmentation
```python
# For imbalanced datasets
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

### 2. Ensemble Methods
```python
# Combine multiple models
from sklearn.ensemble import VotingClassifier
# Train multiple CNN-LSTM models with different parameters
```

### 3. Real-time Detection
```python
# Stream processing capability
def predict_single_connection(connection_features):
    processed_features = detector.preprocess_single(connection_features)
    prediction = detector.model.predict(processed_features)
    return prediction
```

## Troubleshooting Common Issues

### Issue 1: File Format Problems
**Problem**: CSV reading errors
**Solution**: 
```python
# Try different delimiters
data = pd.read_csv('data/KDDTrain+.txt', delimiter=',')
# Or handle missing headers
data = pd.read_csv('data/KDDTrain+.txt', header=None, names=feature_names)
```

### Issue 2: Memory Issues
**Problem**: Out of memory during training
**Solution**:
```python
# Reduce batch size
batch_size = 32

# Use data generators
from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):
    # Implement batch loading
```

### Issue 3: Poor Performance
**Problem**: Low accuracy
**Solution**:
- Check data preprocessing
- Increase model complexity
- Adjust learning rate
- Use different feature selection

### Issue 4: Overfitting
**Problem**: High training accuracy, low validation accuracy
**Solution**:
```python
# Increase dropout
Dropout(0.5)

# Add more regularization
kernel_regularizer=tf.keras.regularizers.l2(0.01)

# Early stopping
EarlyStopping(patience=10)
```

## Performance Optimization Tips

### 1. Hardware Optimization
- **GPU Usage**: Enable CUDA for TensorFlow
- **Memory**: Use 16GB+ RAM for large datasets
- **Batch Size**: Increase if GPU memory allows

### 2. Training Optimization
```python
# Mixed precision training
from tensorflow.keras.mixed_precision import Policy
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.95)
```

### 3. Model Optimization
```python
# Model quantization for deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

## Validation and Testing

### Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5)
cv_scores = []

for train_idx, val_idx in kfold.split(X, y):
    # Train model on fold
    # Evaluate and store scores
```

### Statistical Significance
```python
from scipy import stats
# Compare model performance
t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
```

## Deployment Considerations

### 1. Model Saving and Loading
```python
# Save complete model
model.save('nsl_kdd_cnn_lstm_model.h5')

# Load for inference
loaded_model = tf.keras.models.load_model('nsl_kdd_cnn_lstm_model.h5')
```

### 2. API Development
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(preprocess(data))
    return jsonify({'prediction': int(prediction[0])})
```

### 3. Real-time Monitoring
```python
# Monitor model performance
import mlflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_param("learning_rate", 0.001)
```

This comprehensive guide should help you achieve excellent results with the NSL-KDD dataset using CNN-LSTM architecture!
