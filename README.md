# Deep Learning Course Project: End-to-End DL vs DL-Based Feature Learning
## CAI3105/CS460 - 12th Week Project

**Course**: Deep Learning  
**Institution**: College of Computing and Information Technology – South Valley Campus  
**Lecturer**: Prof. Nashwa El-Bendary  
**Course Code**: CAI3105/CS460  
**Submission Deadline**: Friday, May 8th 2026 (11:55PM)  
**Submission Platform**: Moodle LMS

---

## Project Objective

This project evaluates the performance of **End-to-End Deep Learning (DL) classification** against **DL-based Feature Learning**. It requires a comparative study between:

1. **Approach-1**: A hybrid pipeline that utilizes a pre-trained DL model for feature extraction followed by a traditional Machine Learning (ML) classifier
2. **Approach-2**: A fully integrated end-to-end deep learning model using the same pre-trained DL model

---

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Verify setup
python setup_check.py

# 3. Run the complete project
python main.py
```

## Technical Framework

**PyTorch Implementation** with:
- `torch` & `torchvision` for deep learning
- `torchvision.models` for pre-trained models
- `torch.nn` & `torch.optim` for model building and training
- `scikit-learn` for traditional ML classifiers
- GPU support (CUDA) if available

## Project Structure

```
deep_learning/
├── data/
│   ├── raw/                       # Original datasets
│   │   ├── Brain_MRI/             # Brain tumor detection dataset
│   │   └── Chest_X-Ray/           # Pneumonia detection dataset
│   ├── processed/                 # Preprocessed data (if used)
│   └── augmented/                 # Augmented data (if used)
├── models/                        # Trained model weights
│   ├── Brain-MRI_EfficientNet-B0/
│   ├── Brain-MRI_ResNet50/
│   ├── Chest-XRay_EfficientNet-B0/
│   └── Chest-XRay_ResNet50/
├── scripts/                       # Additional scripts
├── results/
│   ├── plots/                     # Visualization outputs
│   │   ├── Approach-1_confusion_matrix.png
│   │   ├── Approach-2_confusion_matrix.png
│   │   ├── Approach-2_training_history.png
│   │   └── approach_comparison.png
│   ├── metrics/                   # Evaluation metrics
│   │   ├── Approach-1_metrics.txt
│   │   └── Approach-2_metrics.txt
│   └── summary.txt                # Overall project summary
├── documentation/                 # Project documentation
├── config.py                      # Configuration and hyperparameters
├── data_loader.py                 # Data loading and preprocessing
├── approach1_ml_classifier.py     # Approach-1 implementation
├── approach2_end_to_end.py        # Approach-2 implementation
├── utils.py                       # Utility functions
├── download_data.py               # Dataset downloader
├── setup_check.py                 # System verification
├── predict.py                     # Prediction script
├── main.py                        # Main execution script
├── quick_setup.sh                 # Automated setup script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Requirements

### Supported Datasets
- Brain MRI (Medical Imaging)
- Chest X-Ray (Medical Diagnosis)

### Supported Pre-trained Models (PyTorch)
- ResNet50
- EfficientNet-B0

### ML Classifiers (for Approach-1)
- Support Vector Machine (SVM) with linear kernel
- Logistic Regression

---

# REQUIREMENT 1: Dataset Selection and Technical Specifications (5 marks)

## 1.1 Dataset Metadata (1 mark)

The project supports the following publicly available datasets:

| Dataset | Problem Domain | Source | Total Samples |
|---------|---|---|---|
| Chest X-Ray (Pneumonia) | Medical Diagnosis | Kaggle Dataset / Mendeley Data | ~5,856 images |
| Brain MRI (Tumor Detection) | Medical Imaging | Kaggle Dataset / Mendeley Data | ~3,064 images |

**Selected Dataset for this project**: [Brain MRI / Chest X-Ray]

**Source**: [Provide actual source URL]

**Problem Domain**: [Medical Imaging / Medical Diagnosis]

**Total Number of Samples (N)**: [Specify exact count]

## 1.2 Technical Specifications (1 mark)

### Image Resolution
- **Width**: 224 pixels
- **Height**: 224 pixels
- **Format**: RGB or Grayscale (specify for your dataset)

### Color Channels
- **Type**: RGB (3 channels) or Grayscale (1 channel)
- **Bit Depth**: 8-bit per channel

### Number of Classes
- **Class Count**: [Specify number of classes in your dataset]
- **Class Distribution**: [Describe distribution across classes]

**Example for Brain MRI**:
- Resolution: 224×224 pixels
- Channels: Grayscale (1 channel)
- Classes: 4 (glioma, meningioma, notumor, pituitary)


### Preprocessing Steps

1. **Image Resizing**
   - Resize all images to 224×224 pixels using PIL/Pillow
   - Maintains aspect ratio where possible
   - Handles images of varying original sizes

2. **Pixel Normalization**
   - Normalize pixel values to [0, 1] range by dividing by 255
   - Alternative: Use ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
   - Applied consistently across training, validation, and testing sets

3. **Handling Grayscale Images**
   - Repeat single channel to create 3-channel images for pre-trained models
   - Alternative: Train model to accept single-channel input

**Code Reference**: See `data_loader.py` for implementation

## 1.4 Data Augmentation (1 mark)

### Augmentation Techniques Employed

1. **Rotation**: ±20 degrees
   - **Justification**: Increases robustness to image orientation variations commonly found in medical imaging

2. **Width/Height Shifts**: 20%
   - **Justification**: Handles misalignment in image centering

3. **Horizontal Flipping**: Enabled
   - **Justification**: Increases sample diversity without changing class semantics (applicable to medical images)

4. **Zooming**: ±20%
   - **Justification**: Simulates different zoom levels in imaging scenarios

5. **Vertical Flipping**: [Enabled/Disabled]
   - **Justification**: [For medical images, may be disabled to preserve anatomical correctness]

### Necessity and Justification

- **Small Dataset Compensation**: Augmentation artificially increases dataset size, reducing overfitting risk
- **Real-World Variability**: Simulates variations in image capture, orientation, and scale
- **Medical Imaging Context**: Helps models generalize to different imaging conditions and patient positioning

**Code Reference**: See `data_loader.py` for augmentation implementation

## 1.5 Data Splitting (1 mark)

### Train/Validation/Test Split

| Set | Percentage | Number of Images | Purpose |
|-----|-----------|------------------|---------|
| Training | 70% | [Specify count] | Model learning |
| Validation | 15% | [Specify count] | Hyperparameter tuning, early stopping |
| Testing | 15% | [Specify count] | Final evaluation, unseen data assessment |

**Alternative Split**: 80% Training / 20% Testing (if 3-way split not applicable)

### Implementation Strategy
- Stratified split ensuring class distribution preservation
- Random seed fixed for reproducibility
- No data leakage between sets

---

# REQUIREMENT 2: DL Model Selection (4 marks)

## 2.1 Pre-trained Model Selection (1 mark)

**Selected Model**: [ResNet50 / EfficientNet-B0]

### Available Options

| Model | Architecture | Parameters | ImageNet Top-1 Accuracy |
|-------|---|---|---|
| ResNet50 | Residual Networks | 25.5M | 76.13% |
| VGG16 | Visual Geometry Group | 138M | 71.59% |
| EfficientNet-B0 | Efficient Neural Networks | 5.3M | 77.69% |
| MobileNetV1 | Mobile Efficient Networks | 4.2M | 70.42% |

**Selected**: [Model name] - Pre-trained on ImageNet

---

## 2.2 Technical Justification (2 marks)

### Architecture Description and Motivation

#### For ResNet50:
- **Architecture**: Deep Residual Networks with skip connections
- **Layers**: 50 convolutional layers organized in blocks
- **Key Innovation**: Skip connections enable training of very deep networks
- **Advantages**:
  - Proven performance on ImageNet and medical imaging tasks
  - Well-established in literature
  - Strong feature extraction capabilities
- **Why Selected**: Balances depth (feature richness) with computational efficiency

**Architecture Diagram**: [Include diagram from ResNet paper]

```
Input Image → Conv Layer → 
64 filters (7×7, stride 2) → MaxPool → 
[Residual Block ×3 (64 filters)] →
[Residual Block ×4 (128 filters)] →
[Residual Block ×6 (256 filters)] →
[Residual Block ×3 (512 filters)] →
Global Average Pool → Fully Connected (1000) → Output
```

#### For EfficientNet-B0:
- **Architecture**: Efficiently scaled Convolutional Neural Network
- **Key Innovation**: Compound scaling of depth, width, and resolution
- **Advantages**:
  - Smaller model size (5.3M parameters vs 25.5M for ResNet50)
  - Better accuracy-to-efficiency ratio
  - Ideal for resource-constrained environments
- **Why Selected**: Superior performance with minimal computational overhead

**Reference**:
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv preprint arXiv:1512.03385*.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *arXiv preprint arXiv:1905.11946*.

### Transfer Learning Strategy

- **Feature Extraction Layers**: Frozen (weights not updated) for Approach-1
- **Fine-tuning Layers**: Unfrozen for Approach-2 to learn dataset-specific features
- **Rationale**: ImageNet pre-training captures general visual features applicable to medical imaging

---

## 2.3 Hyperparameter Configuration (1 mark)

### Deep Learning Model Hyperparameters (Approach-2)

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Learning Rate** | 0.001 | Standard starting point for Adam optimizer |
| **Batch Size** | 32 | Balances memory usage and training stability |
| **Number of Epochs** | 50 | Sufficient for convergence on medium datasets |
| **Optimizer** | Adam | Adaptive learning rates for robust convergence |
| **Loss Function** | CrossEntropyLoss | Standard for multi-class classification |
| **Early Stopping Patience** | 5 epochs | Prevents overfitting by monitoring validation loss |
| **Weight Decay (L2)** | 1e-5 | Regularization to prevent overfitting |

### ML Classifier Hyperparameters (Approach-1)

#### Support Vector Machine (SVM)

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Kernel** | Linear | Effective for learned feature spaces |
| **C (Regularization)** | 1.0 | Default regularization strength |
| **Max Iterations** | 1000 | Sufficient for convergence |

#### Logistic Regression (Alternative)

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Learning Rate (solver dependent)** | 0.001 | Adaptive learning |
| **Max Iterations** | 1000 | Sufficient iterations for convergence |
| **L2 Penalty** | 0.01 | Regularization to prevent overfitting |

---

# REQUIREMENT 3: Implementation Framework (5 marks)

## 3.1 Approach-1: DL-Based Feature Learning + ML Classifier (3 marks)

### Methodology

**Step 1: Feature Extraction**
```
Input Image (224×224×3) 
  → Pre-trained CNN (frozen weights)
  → Remove classification layer
  → Global Average Pooling (1×1×2048 for ResNet50)
  → Feature Vector (2048-dimensional)
```

**Step 2: ML Classifier Training**
```
Feature Vectors (N × 2048)
  → Normalize features (StandardScaler)
  → Train SVM/Logistic Regression
  → Learn decision boundary
  → Save trained classifier
```

### Implementation Details

1. **Feature Extraction Process**
   - Load pre-trained model (ResNet50 / EfficientNet-B0)
   - Freeze all weights (no gradient updates)
   - Remove final classification layer
   - Process all images through the frozen model
   - Extract penultimate layer activations as features
   - Store features in NumPy arrays or HDF5 format

2. **Normalization**
   - Apply StandardScaler to normalize features
   - Fit scaler on training set only
   - Apply to validation and test sets

3. **ML Classifier Training**
   - Train SVM with linear kernel or Logistic Regression
   - Use scikit-learn implementation
   - Monitor training progress
   - Save trained classifier to disk

### Code Structure

**File**: `approach1_ml_classifier.py`

```python
# Step 1: Load pre-trained model and extract features
feature_extractor = load_pretrained_model('ResNet50', pretrained=True)
freeze_model(feature_extractor)

# Step 2: Extract features for all images
train_features = extract_features(train_loader, feature_extractor)
test_features = extract_features(test_loader, feature_extractor)

# Step 3: Train ML classifier
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

classifier = SVC(kernel='rbf')  # or LogisticRegression()
classifier.fit(train_features_scaled, train_labels)

# Step 4: Evaluate
predictions = classifier.predict(test_features_scaled)
metrics = calculate_metrics(predictions, test_labels)
```

### Performance Evaluation Metrics

| Metric | Definition | Importance |
|--------|-----------|-----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Reliability of positive predictions |
| **Recall** | TP/(TP+FN) | Completeness of positive detection |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Balanced metric for imbalanced datasets |
| **Confusion Matrix** | [TP, FP; FN, TN] | Detailed classification breakdown |

### Expected Output Files

- `results/metrics/Approach-1_metrics.txt`: Classification report with metrics for each class
- `results/plots/Approach-1_confusion_matrix.png`: Visualization of confusion matrix
- `models/[Dataset]_[Model]_approach1_classifier.pkl`: Saved scikit-learn classifier

---

## 3.2 Approach-2: End-to-End Deep Learning Model (2 marks)

### Methodology

**Architecture**:
```
Input Image (224×224×3)
  → Pre-trained CNN Backbone (ResNet50 / EfficientNet-B0)
  → Global Average Pooling
  → Dense Layer (256 units, ReLU)
  → Dropout (0.5)
  → Dense Layer (128 units, ReLU)
  → Dropout (0.5)
  → Classification Layer (num_classes, Softmax)
  → Output Probabilities
```

### Implementation Details

1. **Model Architecture**
   - Load pre-trained model with ImageNet weights
   - Remove final classification layer
   - Add custom head for dataset-specific classification
   - Keep backbone layers unfrozen for fine-tuning

2. **Training Process**
   - Forward pass through network
   - Calculate CrossEntropyLoss
   - Backward pass to compute gradients
   - Update all weights using Adam optimizer
   - Monitor validation metrics for early stopping

3. **Optimization Strategy**
   - Optimizer: Adam (learning_rate=0.001)
   - Loss Function: CrossEntropyLoss
   - Early Stopping: Patience=5 epochs (no improvement in validation loss)
   - Learning Rate Scheduler: Optional (reduce LR if validation loss plateaus)

### Code Structure

**File**: `approach2_end_to_end.py`

```python
# Step 1: Build end-to-end model
model = load_pretrained_model('ResNet50', pretrained=True, num_classes=num_classes)

# Step 2: Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Step 3: Training loop
for epoch in range(num_epochs):
    for batch_images, batch_labels in train_loader:
        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation and early stopping
    val_loss = evaluate_on_validation_set(model, val_loader, criterion)
    if should_early_stop(val_loss, patience=5):
        break

# Step 4: Evaluate on test set
predictions = model(test_images)
metrics = calculate_metrics(predictions, test_labels)
```

### Performance Evaluation Metrics

Same metrics as Approach-1:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class performance breakdown

### Training Visualization

- **Training Curves**: Plot loss and accuracy over epochs
- **Learning Dynamics**: Observe convergence patterns
- **Validation Monitoring**: Detect overfitting (divergence between train and validation metrics)

### Expected Output Files

- `results/metrics/Approach-2_metrics.txt`: Classification report
- `results/plots/Approach-2_confusion_matrix.png`: Confusion matrix visualization
- `results/plots/Approach-2_training_history.png`: Training/validation curves
- `models/[Dataset]_[Model]_approach2_model.pth`: Saved PyTorch model weights

---

# REQUIREMENT 4: Comparative Analysis and Insights (4 marks)

## 4.1 Performance Comparison (2 marks)

### Comparative Analysis Framework

Generate side-by-side comparisons of key metrics:

| Metric | Approach-1 (DL+ML) | Approach-2 (End-to-End) | Difference |
|--------|---|---|---|
| **Accuracy** | [%] | [%] | [+/- %] |
| **Precision** | [%] | [%] | [+/- %] |
| **Recall** | [%] | [%] | [+/- %] |
| **F1-Score** | [%] | [%] | [+/- %] |
| **Training Time** | [seconds] | [seconds] | [+/- %] |
| **Inference Time** | [ms/image] | [ms/image] | [+/- %] |

### Visualization Requirements

**1. Metric Comparison Bar Charts**
- Accuracy, Precision, Recall, F1-Score
- Side-by-side bars for both approaches
- Color coding for better distinction

**2. Confusion Matrices**
- Approach-1 confusion matrix
- Approach-2 confusion matrix
- Heatmap visualization for easy interpretation

**3. Training Curves** (Approach-2 only)
- Loss curves (training vs validation)
- Accuracy curves (training vs validation)
- Highlight convergence and overfitting regions

**4. Efficiency Metrics**
- Training time comparison (bar chart)
- Inference time comparison (milliseconds per image)
- Memory footprint comparison

### Expected Output Visualization

**File**: `results/plots/approach_comparison.png`

```
[Approach-1]        vs        [Approach-2]
┌─────────────┐              ┌─────────────┐
│  Metrics:   │              │  Metrics:   │
│ Acc: 94.5%  │              │ Acc: 96.2%  │
│ Prec: 93.2% │              │ Prec: 95.8% │
│ Rec: 94.1%  │              │ Rec: 96.5%  │
│ F1: 93.6%   │              │ F1: 96.1%   │
└─────────────┘              └─────────────┘

Training Time: 45 sec    Training Time: 180 sec
Inference: 12 ms/img     Inference: 14 ms/img
```

## 4.2 Conclusion and Critical Analysis (2 marks)

Address the following guidance questions:

### Question i: Performance Comparison

**How did the performance of the traditional ML classifier (based on DL-based learned features) compare to the performance of the end-to-end deep learning model?**

**Answer Template**:
- State which approach achieved higher accuracy
- Quantify the difference
- Discuss consistency across other metrics (precision, recall, F1-score)
- Example: "Approach-2 (End-to-End) achieved [X]% accuracy compared to Approach-1's [Y]%, a difference of [Z]%. This suggests that fine-tuning the entire model provides marginal/significant improvement for this task."

### Question ii: Advantages and Limitations

**What are the observed advantages or limitations of using pre-trained DL models as fixed feature extractors for traditional ML algorithms versus allowing the model to learn and classify in a single integrated pipeline?**

**Points to Address**:

**Approach-1 (Feature Extraction + ML) Advantages**:
- Faster training (no backpropagation through deep network)
- Lower computational requirements
- Interpretability of features via ML classifier algorithms
- Easier to implement and debug
- Suitable for resource-constrained environments

**Approach-1 Limitations**:
- Fixed features may not be optimally suited for the target dataset
- Less adaptive to dataset-specific variations
- Potential information loss in pooling layers
- Cannot leverage task-specific learning during feature extraction

**Approach-2 (End-to-End) Advantages**:
- Allows network to adapt features to specific task
- Can achieve better performance through fine-tuning
- Optimization of entire pipeline jointly
- Better feature representation for target classes

**Approach-2 Limitations**:
- Higher computational cost (full backpropagation)
- Risk of overfitting on small datasets
- Longer training time
- Requires more careful hyperparameter tuning

### Question iii: Training Efficiency

**Which approach proved to be more efficient in terms of training time?**

**Analysis Template**:
- Approach-1 (Feature Extraction): O(N) time for feature extraction + training time for ML classifier (usually fast)
- Approach-2 (End-to-End): O(N × E) time where E = number of epochs (typically slower)
- Quantify: "Approach-1 completed training in [X] seconds, while Approach-2 required [Y] seconds, making Approach-1 approximately [Z]× faster."

**Factors Affecting Efficiency**:
- GPU availability and utilization
- Batch size selection
- Number of epochs to convergence
- Early stopping effectiveness

### Question iv: Recommendation for Different Environments

**Based on your findings, which strategy would you recommend for a resource-constrained environment (e.g., mobile or edge computing) versus a high-performance environment?**

**Resource-Constrained Environment (Mobile/Edge)**:
- **Recommendation**: Approach-1 (Feature Extraction + ML)
- **Justification**:
  - Significantly faster inference time
  - Lower memory footprint (no need to store large fine-tuned models)
  - ML classifiers (SVM/LR) are lightweight and efficient
  - Pre-trained features can be compressed or quantized
  - Example: "For a mobile app with limited battery and processing power, Approach-1 would be ideal, reducing inference time by [X]% and memory usage by [Y]%."

**High-Performance Environment (Data Center/Cloud)**:
- **Recommendation**: Approach-2 (End-to-End)
- **Justification**:
  - Superior accuracy for critical applications
  - Computational resources are abundant
  - Training time is less critical
  - Can leverage GPU acceleration fully
  - Example: "For medical diagnostic systems requiring maximum accuracy, Approach-2's [X]% performance gain justifies the additional computational cost."

**Hybrid Recommendation**:
- Use Approach-1 for initial deployment with fast inference
- Use Approach-2 for backend model serving and high-accuracy batch processing
- Consider ensemble approaches combining both methods

---

## Comparative Results Summary

### Generated Outputs

1. **Metrics File**: `results/metrics/Comparison_Summary.txt`
   ```
   Approach-1 vs Approach-2 Comparative Analysis
   ============================================
   
   Accuracy:
     Approach-1: 94.5%
     Approach-2: 96.2%
     Difference: +1.7%
   
   Training Time:
     Approach-1: 45 seconds
     Approach-2: 180 seconds
     Ratio: 4.0×
   
   Recommendation: [Your recommendation based on analysis]
   ```

2. **Visualization Files**:
   - `results/plots/metric_comparison.png`
   - `results/plots/confusion_matrices_comparison.png`
   - `results/plots/training_curves.png`
   - `results/plots/efficiency_comparison.png`

3. **Summary Report**: `results/summary.txt`

---

# BONUS MARKS (+2 Marks)

Students may earn up to **2 bonus marks** by completing any of the following:

## Bonus Option 1: Extra CNN Model (1 Mark)

**Task**: Apply an additional CNN architecture using the same dataset and implement both approaches.

### Implementation Steps

1. **Select Alternative Model**:
   - If you used ResNet50, select EfficientNet-B0 (or vice versa)
   - Or use another approved architecture (VGG16, MobileNetV1)

2. **Run Both Approaches**:
   - Approach-1: Feature extraction + ML classifier
   - Approach-2: End-to-end fine-tuning

3. **Comparative Analysis**:
   - Compare performance between different architectures
   - Analyze trade-offs (accuracy vs inference speed vs model size)
   - Generate comparison visualizations

### Expected Deliverables

| Model | Accuracy | Precision | Recall | F1 | Training Time |
|-------|----------|-----------|--------|----|----|
| ResNet50 (A1) | % | % | % | % | sec |
| ResNet50 (A2) | % | % | % | % | sec |
| EfficientNet-B0 (A1) | % | % | % | % | sec |
| EfficientNet-B0 (A2) | % | % | % | % | sec |

**Analysis**: Discuss which architecture performs best and why, considering both accuracy and computational efficiency.

### Output Files
- `results/plots/multi_model_comparison.png`
- `results/metrics/Multi_Model_Analysis.txt`

---

## Bonus Option 2: Extra Dataset (1 Mark)

**Task**: Evaluate the primary Deep Learning model on a second dataset.

### Implementation Steps

1. **Select Second Dataset**:
   - If you used Brain MRI, evaluate on Chest X-Ray
   - Or use another approved dataset (CIFAR-10, Plant Village)

2. **Run Both Approaches**:
   - Apply the same model architecture to new dataset
   - Maintain consistent hyperparameters for fair comparison

3. **Cross-Dataset Analysis**:
   - Analyze model generalization
   - Discuss domain adaptation challenges
   - Compare performance metrics

### Expected Deliverables

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---------|-------|----------|-----------|--------|----| 
| Brain MRI | ResNet50 (A1) | % | % | % | % |
| Brain MRI | ResNet50 (A2) | % | % | % | % |
| Chest X-Ray | ResNet50 (A1) | % | % | % | % |
| Chest X-Ray | ResNet50 (A2) | % | % | % | % |

**Analysis**: Discuss generalization performance, domain differences, and applicability across medical imaging tasks.

### Output Files
- `results/plots/cross_dataset_comparison.png`
- `results/metrics/Cross_Dataset_Analysis.txt`

---

## Bonus: Completing Both Options (2 Marks)

Combine both bonus options for maximum impact:
- Multiple models on multiple datasets
- Comprehensive comparative analysis table
- Deep insights into model architecture selection and domain adaptation
- Demonstrating advanced experimental design

---

## Installation and Setup

### 1. Clone or Navigate to Project Directory
```bash
cd ~/deep_learning
```

### 2. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python setup_check.py
```

### 5. Configure Kaggle Credentials (Optional - for Kaggle datasets)

For Brain MRI, Chest X-Ray, or Plant Village datasets:

1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`
5. Verify: `kaggle datasets list`

---

## Dataset Download and Setup

### Automatic Download (Recommended)

Simply run:
```bash
python main.py
```

The script will automatically check for datasets and prompt you to download if needed.

### Manual Dataset Download

For more control, use:
```bash
python download_data.py
```

This provides an interactive menu to:
- Download CIFAR-10 (~180 MB) - No credentials needed
- Download Brain MRI - Requires Kaggle credentials
- Download Chest X-Ray - Requires Kaggle credentials  
- Download Plant Village - Requires Kaggle credentials (~3.5 GB)
- Setup Kaggle API credentials
- Verify existing downloads

### Dataset Storage Requirements

| Dataset | Size | Storage |
|---------|------|---------|
| CIFAR-10 | ~180 MB | 200 MB |
| Brain MRI | ~2 GB | 2.5 GB |
| Chest X-Ray | ~1.5 GB | 2 GB |
| Plant Village | ~3.5 GB | 4 GB |

**Total space recommended**: 10 GB minimum

---

## Configuration

### Main Configuration File: `config.py`

Edit `config.py` to customize:

```python
# Dataset Configuration
DATASET_CONFIG = {
    'name': 'Brain-MRI',  # or 'Chest-XRay', 'CIFAR-10', 'Plant-Village'
    'data_dir': './data/raw',
    'image_size': (224, 224),
    'num_classes': 4,  # Brain MRI: 4 classes
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
}

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'EfficientNet-B0',  # or 'ResNet50'
    'pretrained': True,
    'freeze_backbone': False,  # For Approach-2
}

# Approach-1 ML Classifier
APPROACH1_CONFIG = {
    'classifier_type': 'svm',  # or 'logistic_regression'
    'svm_kernel': 'linear',
    'svm_C': 1.0,
}

# Training Configuration (Approach-2)
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'early_stopping_patience': 5,
    'weight_decay': 1e-5,
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.2,
}

# GPU Settings
GPU_CONFIG = {
    'use_gpu': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```

---

## Usage

### Step 1: Download and Prepare Data

```bash
# Interactive menu for dataset selection
python download_data.py
```

Or let `main.py` prompt you:
```bash
python main.py  # Will ask if dataset is missing
```

### Step 2: Configure Project Settings

Edit `config.py`:
- Select dataset (Brain-MRI, Chest-XRay, CIFAR-10)
- Choose model architecture (ResNet50, EfficientNet-B0)
- Set hyperparameters

### Step 3: Run Complete Project

```bash
python main.py
```

**What main.py does**:
1. ✅ Checks for dataset (auto-downloads if needed)
2. ✅ Loads and preprocesses data with augmentation
3. ✅ Trains Approach-1 (Feature Extraction + ML Classifier)
4. ✅ Trains Approach-2 (End-to-End DL Model)
5. ✅ Generates comparative visualizations
6. ✅ Saves results and metrics
7. ✅ Displays summary report

### Step 4: Make Predictions on New Images

```bash
python predict.py --image_path path/to/image.jpg --approach 2
```

---

## Output Files and Results

### Generated Outputs

After running `main.py`, the following files will be created:

#### Metrics Files
```
results/metrics/
├── Approach-1_metrics.txt       # Classification report for ML classifier
├── Approach-2_metrics.txt       # Classification report for End-to-End model
└── Comparison_Summary.txt       # Side-by-side comparison
```

#### Visualization Files
```
results/plots/
├── Approach-1_confusion_matrix.png        # Confusion matrix for Approach-1
├── Approach-2_confusion_matrix.png        # Confusion matrix for Approach-2
├── Approach-2_training_history.png        # Training/validation curves
├── metric_comparison.png                  # Bar charts of metrics
├── confusion_matrices_comparison.png      # Side-by-side confusion matrices
├── efficiency_comparison.png              # Training time comparison
└── approach_comparison.png                # Overall comparison
```

#### Model Files
```
models/
├── [Dataset]_[Model]_approach1_classifier.pkl   # Trained ML classifier
└── [Dataset]_[Model]_approach2_model.pth        # Fine-tuned PyTorch model
```

#### Summary Report
```
results/summary.txt        # Overall project summary with key findings
```

### Example Metrics Output

**File**: `results/metrics/Approach-2_metrics.txt`
```
Classification Report:
==================

               precision    recall  f1-score   support

      class_0       0.95      0.96      0.96       250
      class_1       0.93      0.92      0.92       240
      class_2       0.94      0.95      0.94       255
      class_3       0.96      0.94      0.95       255

    accuracy                           0.944      1000
   macro avg       0.95      0.94      0.94      1000
weighted avg       0.95      0.94      0.94      1000

Confusion Matrix:
[240   5   3   2]
[  4 221   8   7]
[  2   6 242   5]
[  3   8   4 240]

Overall Accuracy: 94.4%
```

---

## Advanced Features and Options

### Early Stopping (Approach-2)

Prevents overfitting by monitoring validation loss:
- Patience: 5 epochs
- Restored best model after training stops
- Automatically enabled in `main.py`

### Fine-tuning Strategy

By default, Approach-2 fine-tunes all layers. To freeze certain layers:

**In config.py**:
```python
MODEL_CONFIG = {
    'freeze_backbone': True,      # Freeze pre-trained layers
    'num_fine_tune_layers': 2,    # Only fine-tune last 2 blocks
}
```

### Learning Rate Scheduling (Optional)

Enable in `config.py`:
```python
TRAINING_CONFIG = {
    'use_lr_scheduler': True,
    'scheduler_type': 'exponential',  # or 'step', 'cosine'
    'scheduler_gamma': 0.1,
}
```

### GPU Acceleration

Automatic GPU detection and usage:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
CUDA_VISIBLE_DEVICES="" python main.py

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python main.py
```

### Inference on Custom Dataset

**File**: `predict.py`

```bash
# Single image
python predict.py --image_path data/test_image.jpg --approach 2

# Directory of images
python predict.py --image_dir data/custom_images/ --approach 1

# Batch predictions
python predict.py --image_dir data/ --batch_size 32 --output results/predictions.csv
```

---

## Troubleshooting

### Dataset Issues

**CIFAR-10 Won't Download**
```bash
# Check directory permissions
chmod -R 755 ./data/

# Manual download
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data/raw/', download=True)"
```

**Kaggle Dataset Errors**
- Verify `~/.kaggle/kaggle.json` exists and is readable
- Run: `chmod 600 ~/.kaggle/kaggle.json`
- Test credentials: `kaggle datasets list`

**Disk Space Issues**
- Ensure 10 GB free space
- Check: `df -h`
- Delete unused datasets: `rm -rf data/raw/[unwanted_dataset]`

### Memory and Performance

**Out of Memory (OOM)**
```python
# In config.py, reduce batch size
TRAINING_CONFIG = {'batch_size': 16}  # Reduced from 32
```

**Slow Training**
- Verify GPU usage: `nvidia-smi`
- Use EfficientNet-B0 instead of ResNet50 (faster)
- Reduce number of epochs
- Use smaller image size (e.g., 224 → 192)

### GPU/PyTorch Issues

**CUDA Not Available**
```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Update PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**PyTorch Version Mismatch**
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install -r requirements.txt
```

### Model Loading Issues

**Pre-trained Weights Not Found**
```bash
# PyTorch will auto-download, or manually
python -c "from torchvision.models import resnet50; resnet50(pretrained=True)"
```

---

## Project References

### Deep Learning Architecture Papers

1. **ResNet50**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
   - https://arxiv.org/abs/1512.03385

2. **EfficientNet-B0**: Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *arXiv preprint arXiv:1905.11946*.
   - https://arxiv.org/abs/1905.11946

3. **VGG16**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *arXiv preprint arXiv:1409.1556*.
   - https://arxiv.org/abs/1409.1556

4. **MobileNetV1**: Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv preprint arXiv:1704.04861*.
   - https://arxiv.org/abs/1704.04861

### Machine Learning Methods

- **Support Vector Machines**: Cortes, C., & Vapnik, V. (1995). "Support-vector networks." *Machine Learning*, 20(3), 273-297.
- **Logistic Regression**: Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." *Journal of the Royal Statistical Society*, 20(2), 215-242.

### Deep Learning Frameworks

- **PyTorch Official Documentation**: https://pytorch.org/docs
- **Torchvision Models**: https://pytorch.org/vision/stable/models.html
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html

### Related Tutorials and Resources

- Transfer Learning with PyTorch: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- CNN Feature Extraction: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- ImageNet Pre-trained Models: https://github.com/pytorch/vision/blob/main/torchvision/models/__init__.py

---

## Submission Requirements and Checklist

### Team Submission (Each Member Must Submit)

- [ ] **1. PDF Report** (via Moodle LMS)
  - [ ] Requirement 1: Dataset Selection and Technical Specifications (5 marks)
  - [ ] Requirement 2: DL Model Selection (4 marks)
  - [ ] Requirement 3: Implementation Framework (5 marks)
  - [ ] Requirement 4: Comparative Analysis and Insights (4 marks)
  - [ ] Bonus Marks (up to +2 marks)
  - [ ] Formal structure and professional formatting

- [ ] **2. GitHub Repository Link** (via Moodle LMS)
  - [ ] Public repository with complete, runnable code
  - [ ] All scripts: data preprocessing, DL feature extraction, ML classification
  - [ ] README.md with setup instructions
  - [ ] Data download scripts
  - [ ] Model training scripts
  - [ ] Results and visualizations
  - [ ] License information

- [ ] **3. One-Minute Video Presentation** (YouTube link via Moodle LMS)
  - [ ] Brief dataset and architecture overview
  - [ ] Summary of primary results comparing approaches
  - [ ] Personal technical insights about pre-trained models
  - [ ] Duration: ~1 minute
  - [ ] Upload to personal YouTube channel
  - [ ] Submit shareable link on Moodle

- [ ] **4. A3 Poster** (Printed for in-class presentation)
  - [ ] Follow provided template
  - [ ] Visually compelling layout
  - [ ] Key results and comparative analysis
  - [ ] Bring to Saturday presentation

### In-Class Presentation (Per Team)

- [ ] **Date & Time**: Saturday, May 9th 2026, during scheduled lecture
- [ ] **Duration**: 7-10 minutes per team
- [ ] **Dress Code** (1 mark): Formal or semi-formal attire
- [ ] **A3 Poster** (1 mark): Printed and displayed
- [ ] **Technical Readiness**: Laptop with dataset and runnable code
- [ ] **Presentation Content**:
  - [ ] Dataset introduction
  - [ ] Architecture justification
  - [ ] Results comparison
  - [ ] Insights and conclusions
  - [ ] Q&A readiness

### Bonus Opportunities

- [ ] **Extra CNN Model** (1 mark): Apply additional architecture
- [ ] **Extra Dataset** (1 mark): Evaluate on second dataset
- [ ] **Comprehensive Analysis**: Present comparative findings for bonus options

---

## Running the Project Step-by-Step

### Complete Workflow

```bash
# Step 1: Setup Environment
cd ~/deep_learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 2: Verify Installation
python setup_check.py

# Step 3: Download Dataset
python download_data.py
# Select your dataset from the interactive menu

# Step 4: Configure Settings
# Edit config.py to select:
# - Dataset name
# - Model architecture
# - Hyperparameters
# - ML classifier type

# Step 5: Run Complete Project
python main.py
# This will automatically:
# - Load and preprocess data
# - Train Approach-1 (Feature Extraction + ML)
# - Train Approach-2 (End-to-End DL)
# - Generate visualizations and metrics
# - Save all results

# Step 6: Review Results
# Examine outputs in:
# - results/metrics/
# - results/plots/
# - results/summary.txt

# Step 7: Generate Reports (Optional)
python predict.py --image_dir data/test/ --output results/test_predictions.csv
```

### Expected Timeline

| Step | Time | Task |
|------|------|------|
| 1 | 5 min | Environment setup |
| 2 | 2 min | Verify installation |
| 3 | 10-30 min | Download dataset (depends on dataset size) |
| 4 | 1 min | Configure settings |
| 5 | 30-60 min | Run complete project (depends on GPU/CPU) |
| 6 | 5 min | Review results |
| **Total** | **1-2 hours** | Complete run (first time) |

---

## Key Performance Benchmarks

### Actual Project Results

Comprehensive comparison across multiple models and datasets:

#### Brain-MRI Dataset with EfficientNet-B0
| Approach | Accuracy | F1-Score | Training Time |
|----------|----------|----------|---|
| Approach-1 (Feature Extraction + ML) | **93.89%** | 0.9392 | 14.64s |
| Approach-2 (End-to-End DL) | **97.31%** | 0.9733 | 359.67s |
| **Difference** | **+3.42%** | +0.0341 | ~345s |

#### Brain-MRI Dataset with ResNet50
| Approach | Accuracy | F1-Score | Training Time |
|----------|----------|----------|---|
| Approach-1 (Feature Extraction + ML) | **92.13%** | 0.9213 | 28.91s |
| Approach-2 (End-to-End DL) | **94.26%** | 0.9423 | 992.28s |
| **Difference** | **+2.13%** | +0.0210 | ~963s |

#### Chest X-Ray Dataset with EfficientNet-B0
| Approach | Accuracy | F1-Score | Training Time |
|----------|----------|----------|---|
| Approach-1 (Feature Extraction + ML) | **92.72%** | 0.9267 | 10.02s |
| Approach-2 (End-to-End DL) | **95.22%** | 0.9530 | 214.18s |
| **Difference** | **+2.50%** | +0.0263 | ~204s |

#### Chest X-Ray Dataset with ResNet50
| Approach | Accuracy | F1-Score | Training Time |
|----------|----------|----------|---|
| Approach-1 (Feature Extraction + ML) | **93.52%** | 0.9348 | 18.10s |
| Approach-2 (End-to-End DL) | **96.25%** | 0.9623 | 500.84s |
| **Difference** | **+2.73%** | +0.0275 | ~483s |

### Performance Summary

**Best Overall Accuracy**: Brain-MRI with EfficientNet-B0 Approach-2 (**97.31%**)

**Key Observations**:
1. Approach-2 (End-to-End) consistently outperforms Approach-1 across all datasets and models
2. Accuracy improvement ranges from 2.13% to 3.42%
3. EfficientNet-B0 provides better accuracy-to-training-time ratio than ResNet50
4. Approach-1 is significantly faster (10-29s vs 214-992s for Approach-2)
5. For medical imaging tasks, the accuracy gains of Approach-2 justify the additional training time

### Expected Results (Reference Benchmarks)

These are typical ranges for similar datasets and models:

#### Brain MRI Dataset
- **Approach-1 Accuracy**: 85-92% (Actual: 92-94%)
- **Approach-2 Accuracy**: 90-96% (Actual: 94-97%)
- **Training Time (Approach-1)**: 2-5 minutes (Actual: 15-29s)
- **Training Time (Approach-2)**: 10-30 minutes (Actual: 6-17 minutes)

#### Chest X-Ray Dataset
- **Approach-1 Accuracy**: 90-95% (Actual: 93-94%)
- **Approach-2 Accuracy**: 92-97% (Actual: 95-96%)
- **Training Time (Approach-1)**: 3-8 minutes (Actual: 10-18s)
- **Training Time (Approach-2)**: 15-40 minutes (Actual: 3.5-8 minutes)

#### CIFAR-10 Dataset
- **Approach-1 Accuracy**: 75-85%
- **Approach-2 Accuracy**: 80-90%
- **Training Time (Approach-1)**: 5-15 minutes
- **Training Time (Approach-2)**: 20-60 minutes

**Note**: Performance depends on:
- Model architecture (ResNet50 vs EfficientNet-B0)
- Hyperparameter tuning
- Data augmentation effectiveness
- Hardware capabilities (GPU vs CPU)
- Training duration allowed

---

### Code Statistics

| File | Lines of Code | Purpose |
|------|---|---|
| `main.py` | ~500-700 | Main execution orchestration |
| `data_loader.py` | ~300-400 | Data loading and preprocessing |
| `approach1_ml_classifier.py` | ~200-300 | Feature extraction + ML training |
| `approach2_end_to_end.py` | ~300-500 | End-to-end DL training |
| `config.py` | ~100-150 | Configuration parameters |
| `utils.py` | ~200-300 | Utility functions |
| `predict.py` | ~100-200 | Inference on new images |
| **Total** | **~1500-2400 lines** | Complete project |

### Dataset Statistics

| Dataset | Train Images | Val Images | Test Images | Classes | Size |
|---------|---|---|---|---|---|
| CIFAR-10 | 42,000 | 9,000 | 9,000 | 10 | ~180 MB |
| Brain MRI | 2,100 | 465 | 465 | 4 | ~2 GB |
| Chest X-Ray | 4,095 | 906 | 906 | 2 | ~1.5 GB |
| Plant Village | 38,100 | 8,415 | 8,415 | 38 | ~3.5 GB |

---

## Final Notes

### Grading Criteria Summary

| Component | Marks | Weight |
|-----------|-------|--------|
| Requirement 1: Dataset & Specs | 5 | 25% |
| Requirement 2: Model Selection | 4 | 20% |
| Requirement 3: Implementation | 5 | 25% |
| Requirement 4: Analysis & Insights | 4 | 20% |
| **Total Base Marks** | **20** | **100%** |
| **Bonus (Optional)** | **+2** | **Extra Credit** |

### Submission Timeline

- **Coding Deadline**: May 6th, 2026
- **Submission Deadline**: May 8th, 2026 (11:55PM)
- **In-Class Presentation**: May 9th, 2026

### Important Notes

1. **Team Collaboration**: While working collaboratively, each team member must submit individual reports and presentations to Moodle.

2. **Code Quality**: Ensure code is well-documented, properly commented, and follows Python PEP8 standards.

3. **Reproducibility**: Include random seeds and environment specifications so results can be reproduced.

4. **Time Management**: Start early! Training deep learning models can take significant time, especially on CPU.

5. **Save Checkpoints**: Regularly save model checkpoints during training to avoid losing progress.

6. **Document Assumptions**: Clearly state any assumptions or modifications made to the project requirements.

7. **Backup Data**: Keep backups of your dataset and trained models to avoid data loss.

---

## Frequently Asked Questions

**Q: Can I use a different pre-trained model?**  
A: Yes! Models like VGG16 and MobileNetV1 are supported. Modify `config.py` to select them.

**Q: Can I use a different ML classifier?**  
A: Yes! You can implement Decision Trees, Random Forests, etc. Modify `approach1_ml_classifier.py`.

**Q: Can I use a different dataset?**  
A: Yes! Implement a custom data loader in `data_loader.py` following the same structure.

**Q: How do I optimize for GPU?**  
A: PyTorch automatically uses GPU if available. Ensure you have PyTorch-CUDA installed.

**Q: What if I don't have GPU?**  
A: The code runs on CPU but will be slower. Reduce batch size or epochs to speed up training.

**Q: How do I handle class imbalance?**  
A: Use stratified splitting, weighted loss functions, or data augmentation techniques.

**Q: Can I use transfer learning from other sources?**  
A: The project focuses on ImageNet pre-trained models. Using other sources requires documentation and justification.

---

## Contact and Support

For questions or issues:
1. Check the Troubleshooting section
2. Review the FAQ above
3. Check PyTorch and scikit-learn official documentation
4. Contact your instructor during office hours
5. Consult with your team members

---

## License and Attribution

**Project License**: Academic Use - Course Project  
**Course**: Deep Learning (CAI3105/CS460)  
**Institution**: College of Computing and Information Technology – South Valley Campus  
**Lecturer**: Prof. Nashwa El-Bendary  
**Department**: Computer Science

This project is designed for educational purposes to demonstrate deep learning techniques and comparative analysis methodologies.

---

## Acknowledgments

This project incorporates concepts and best practices from:
- PyTorch official tutorials and documentation
- Computer vision and deep learning literature
- Best practices in machine learning experimentation
- Academic research in transfer learning and feature extraction

---

**Good luck with your project! 🚀**

For questions, contact your course instructor or teaching assistant.

**Last Updated**: May 2026  
**Version**: 2.0 (Comprehensive Requirements Edition)  
**Framework**: PyTorch  
**Submission Deadline**: Friday, May 8th 2026 (11:55PM)  
**Presentation Date**: Saturday, May 9th 2026

