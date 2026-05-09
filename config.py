

DATASETS_CONFIG = [
    {
        'name': 'Brain-MRI',
        'image_size': (128, 128),
        'color_channels': 3,
        'num_classes': 4,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'path': './data/brain_mri'
    },
    {
        'name': 'Chest-XRay',
        'image_size': (128, 128),
        'color_channels': 3,  # هنحول grayscale → 3 channels
        'num_classes': 2,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'path': './data/chest_xray'
    }]
MODELS_CONFIG = [
    {
        'architecture': 'EfficientNet-B0',
        'pretrained': True
    },
    {
        'architecture': 'ResNet50',
        'pretrained': True
    },
   
]
# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.2,
}

TRAINING_CONFIG = {
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'cross_entropy',
    'validation_split': 0.2,
    'early_stopping': True,
    'early_stopping_patience': 5,
}
# ML Classi
# fier Configuration - Approach 1
ML_CLASSIFIER_CONFIG = {
    'classifier_type': 'SVM',  
    'svm_kernel': 'rbf', 
    'svm_C': 1.0,
    'lr_max_iter': 1000,
    'test_size': 0.2,
}

# File Paths
PATHS = {
    'raw_data': './data/raw/',
    'processed_data': './data/processed/',
    'augmented_data': './data/augmented/',
    'models': './models/',
    'results': './results/',
    'plots': './results/plots/',
    'metrics': './results/metrics/',
}

# Evaluation Metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
