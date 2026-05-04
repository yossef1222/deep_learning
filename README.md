# Deep Learning Course Project: End-to-End DL vs DL+ML Feature Extraction

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Verify setup
python setup_check.py

# 3. Download datasets (or let main.py auto-prompt)
python download_data.py

# 4. Run the complete project
python main.py
```

## Project Overview

This project evaluates the performance of two deep learning strategies for image classification using **PyTorch**:

1. **Approach-1 (Hybrid Pipeline)**: Deep Learning-based Feature Extraction + Traditional ML Classifier
   - Uses a pre-trained CNN as a fixed feature extractor
   - Extracts feature vectors from images
   - Trains a traditional ML classifier (SVM or Logistic Regression)

2. **Approach-2 (End-to-End)**: Fully Integrated Deep Learning Model
   - Uses the same pre-trained CNN architecture
   - Fine-tunes the model end-to-end for classification

## Framework

**PyTorch Implementation** with:
- `torch` & `torchvision` for deep learning
- `torchvision.models` for pre-trained models
- `torch.nn` & `torch.optim` for model building and training
- GPU support (CUDA) if available

## Project Structure

```
deep_learning/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed data
│   └── augmented/           # Augmented data
├── models/                  # Trained model weights
├── scripts/                 # Additional scripts (optional)
├── results/
│   ├── plots/               # Visualization outputs
│   └── metrics/             # Evaluation metrics
├── documentation/           # Project documentation
├── config.py                # Configuration and hyperparameters
├── data_loader.py           # Data loading and preprocessing (PyTorch)
├── approach1_ml_classifier.py  # Approach-1 (PyTorch feature extraction + sklearn)
├── approach2_end_to_end.py  # Approach-2 (PyTorch end-to-end training)
├── utils.py                 # Utility functions
├── download_data.py         # Dataset downloader (interactive menu)
├── setup_check.py           # System verification script
├── main.py                  # Main execution script
├── quick_setup.sh           # Automated setup script (Linux/Mac)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Requirements

### Supported Datasets
- CIFAR-10 (General Object Recognition)
- Brain MRI (Medical Imaging)
- Chest X-Ray (Medical Diagnosis)
- Plant Village (Agriculture)

### Supported Pre-trained Models (PyTorch)
- ResNet50
- EfficientNet-B0

### ML Classifiers (for Approach-1)
- Support Vector Machine (SVM)
- Logistic Regression

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd ~/deep_learning
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Datasets** (Optional - will auto-prompt on first run)
   ```bash
   # Interactive dataset selection
   python download_data.py
   
   # Or let main.py prompt you when running:
   python main.py
   ```

## Data Setup

### Quick Start (Recommended)
Simply run:
```bash
python main.py
```
The script will automatically check for datasets and prompt you to download if needed.

### Manual Dataset Download
For more control, use the dedicated download script:
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

### Kaggle Datasets (Optional)
For Kaggle datasets, you'll need API credentials:

1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

Then use `download_data.py` to download Brain MRI, Chest X-Ray, or Plant Village datasets.

## Configuration

Edit `config.py` to set:
- **Dataset name** (CIFAR-10, Brain-MRI, etc.)
- **Model architecture** (ResNet50, VGG16, EfficientNet-B0, MobileNetV2)
- **Hyperparameters** (learning rate, batch size, epochs, etc.)
- **ML classifier type** (SVM or Logistic Regression)
- **Data augmentation settings**

### Example Configuration
```python
DATASET_CONFIG = {
    'name': 'CIFAR-10',
    'image_size': (224, 224),
    'num_classes': 10,
    'train_split': 0.7,
}

MODEL_CONFIG = {
    'architecture': 'EfficientNet-B0',  # or 'ResNet50'
    'pretrained': True,
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'adam',
}
```

## Usage

### Download Datasets

```bash
# Interactive menu for dataset selection
python download_data.py
```

### Run the Complete Project

```bash
# Automatically checks for data and prompts if needed
python main.py
```

This will:
1. Check if dataset exists (auto-download if needed)
2. Load and preprocess the dataset
3. Train Approach-1 (DL Feature Extraction + ML) using PyTorch
4. Train Approach-2 (End-to-End DL) using PyTorch
5. Generate comparative analysis plots
6. Save results and metrics

### Output Files

After running `main.py`, the following files will be generated:

- `results/plots/`:
  - `Approach-1_confusion_matrix.png` - Confusion matrix for Approach-1
  - `Approach-2_confusion_matrix.png` - Confusion matrix for Approach-2
  - `Approach-2_training_history.png` - Training curves (loss and accuracy)
  - `approach_comparison.png` - Side-by-side comparison of both approaches

- `results/metrics/`:
  - `Approach-1_metrics.txt` - Classification report for Approach-1
  - `Approach-2_metrics.txt` - Classification report for Approach-2

- `results/summary.txt` - Overall project summary

## Key Results

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Performance Comparison
The project generates visualizations comparing:
- Accuracy, Precision, Recall, and F1-Score
- Training time
- Confusion matrices
- Training curves (loss and accuracy over epochs)

## Experimental Design

### Core Experiment
- Single dataset (e.g., CIFAR-10)
- Single model architecture (e.g., EfficientNet-B0)
- Comparison of two approaches

### Extended Study (Optional)
For more comprehensive evaluation, modify `config.py` to test:
- Multiple model architectures (ResNet50, EfficientNet-B0)
- Multiple datasets (CIFAR-10, Brain-MRI)
- Impact of different hyperparameters

## Data Preprocessing

1. **Resizing**: All images resized to 224×224 using PIL
2. **Normalization**: Pixel values normalized to [0, 1]
3. **Augmentation**: Applied techniques include:
   - Rotation (20°)
   - Width/Height shifts (20%)
   - Horizontal flipping
   - Zooming (20%)
4. **PyTorch Format**: Data converted to tensors with shape (N, C, H, W)

## Model Details

### Approach-1: Feature Extraction + ML (PyTorch)
```
Input Image → Pre-trained CNN (frozen) → Feature Extraction → 
Flatten Features → Normalize → ML Classifier (SVM/LR) → Output
```

### Approach-2: End-to-End (PyTorch)
```
Input Image → Pre-trained CNN (fine-tuned) → Global Average Pooling → 
Dense Layers (256 units) → Output Classification Layer → Output
```

## Hyperparameters

### Deep Learning (Approach-2, PyTorch)
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 50
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

### ML Classifier (Approach-1)
- SVM Kernel: Linear
- SVM C: 1.0
- Logistic Regression Max Iterations: 1000

## Advanced Features

### Early Stopping
Prevents overfitting by monitoring validation loss (patience: 5 epochs)

### Fine-tuning (Optional)
Uncomment in `main.py` to fine-tune the base model with a lower learning rate

### GPU Acceleration
- Automatic GPU detection using `torch.cuda.is_available()`
- Models and data automatically moved to GPU/CPU as needed
- Set `CUDA_VISIBLE_DEVICES` environment variable to control GPU usage

### Custom Datasets
Modify `data_loader.py` to load custom datasets in the expected format

## Troubleshooting

### Dataset Download Issues

**CIFAR-10 Won't Download**
- Check internet connection
- Ensure `/data/raw/` directory exists and is writable
- Try manually: `python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data/raw/', download=True)"`

**Kaggle Dataset Download Issues**
- Ensure `kaggle.json` is in `~/.kaggle/`
- Run: `chmod 600 ~/.kaggle/kaggle.json`
- Verify credentials: `kaggle datasets list` should show available datasets
- For detailed setup: https://github.com/Kaggle/kaggle-api#installation

**Permission Denied**
- Run: `chmod -R 755 ./data/`

**Disk Space Issues**
- CIFAR-10: ~180 MB
- Brain MRI: ~2 GB
- Chest X-Ray: ~1.5 GB
- Plant Village: ~3.5 GB
- Ensure sufficient free space before downloading

### Memory Issues
- Reduce `batch_size` in `config.py`
- Use `EfficientNet-B0` or `MobileNetV2` (smaller models)

### Dataset Not Found
- Ensure dataset is in the correct location
- Check `PATHS` configuration in `config.py`

### GPU/CPU Selection
- PyTorch automatically uses GPU if available
- To force CPU: Set `CUDA_VISIBLE_DEVICES=""` before running
- To use specific GPU: Set `CUDA_VISIBLE_DEVICES=0` for GPU 0

### PyTorch Issues
- Update PyTorch: `pip install --upgrade torch torchvision`
- Check CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

## Project Requirements (Course)

✅ **Requirement 1**: Dataset Selection and Technical Specifications
✅ **Requirement 2**: DL Model Selection with justification
✅ **Requirement 3**: Implementation Framework (both approaches with PyTorch)
✅ **Requirement 4**: Comparative Analysis and Insights

## Submission Checklist

- [ ] PDF Report with all requirements
- [ ] GitHub repository link with complete code
- [ ] 1-minute video presentation
- [ ] A3 Poster for in-class presentation
- [ ] Runnable code with dataset
- [ ] Results and visualizations

## References

- PyTorch Official: https://pytorch.org
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- Scikit-learn: https://scikit-learn.org
- ImageNet Pre-trained Models: https://en.wikipedia.org/wiki/ImageNet

## License

Academic Use - Course Project

## Author

Student Name - Deep Learning Course (CAI3105/CS460)

---

**Framework**: PyTorch  
**Submission Deadline**: Thursday, May 7th 2026 (11:55PM)  
**Presentation Date**: Friday, May 8th 2026

