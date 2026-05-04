"""
Data loading and preprocessing module
Handles dataset loading, preprocessing, augmentation, and splitting
"""

import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
from pathlib import Path
import os


class DataLoader:
    """Handle data loading and preprocessing"""
    
    def __init__(self, config):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary from config.py
        """
        self.config = config
        self.dataset_config = config['DATASET_CONFIG']
        self.augmentation_config = config['AUGMENTATION_CONFIG']
        self.image_size = self.dataset_config['image_size']
        self.num_classes = self.dataset_config['num_classes']
    
    def load_cifar10(self):
        """Load CIFAR-10 dataset"""
        import torchvision.datasets as datasets
        
        print("Loading CIFAR-10 dataset...")
        
        # Check if already downloaded
        cifar_path = Path('./data/raw/cifar-10-batches-py')
        if not cifar_path.exists():
            try:
                cifar_train = datasets.CIFAR10(root='./data/raw/', train=True, download=True)
                cifar_test = datasets.CIFAR10(root='./data/raw/', train=False, download=True)
            except Exception as e:
                print(f"\n✗ Failed to download CIFAR-10: {str(e)}")
                print("\nPlease download manually:")
                print("  1. Run: python download_data.py")
                print("  2. Or visit: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
                raise
        else:
            cifar_train = datasets.CIFAR10(root='./data/raw/', train=True, download=False)
            cifar_test = datasets.CIFAR10(root='./data/raw/', train=False, download=False)
        
        # Convert to numpy arrays
        X_train = np.array([np.array(img) for img, _ in cifar_train])
        y_train = np.array([label for _, label in cifar_train])
        
        X_test = np.array([np.array(img) for img, _ in cifar_test])
        y_test = np.array([label for _, label in cifar_test])
        
        # Combine and reshuffle
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        
        print(f"✓ Loaded {len(X)} samples")
        return X, y
    
    def load_brain_mri(self, data_path='./data/raw/Brain_MRI'):
        """
        Load Brain MRI dataset
        Handles Training/Testing folder structure with class subfolders
        """
        from PIL import Image
        
        print("Loading Brain MRI dataset...")
        images = []
        labels = []
        
        # Map class names to indices
        class_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        
        # Load from both Training and Testing folders
        for folder_type in ['Training', 'Testing']:
            folder_path = os.path.join(data_path, folder_type)
            if not os.path.exists(folder_path):
                continue
            
            for class_name, class_idx in class_map.items():
                class_path = os.path.join(folder_path, class_name)
                if not os.path.exists(class_path):
                    continue
                
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize(self.image_size)
                            images.append(np.array(img))
                            labels.append(class_idx)
                        except Exception as e:
                            print(f"  Warning: Could not load {img_file}: {str(e)}")
        
        X = np.array(images)
        y = np.array(labels)
        
        if len(X) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        print(f"✓ Loaded {len(X)} samples from {len(np.unique(y))} classes")
        return X, y
    
    def load_chest_xray(self, data_path='./data/raw/Chest_X-Ray'):
        """
        Load Chest X-Ray dataset
        Handles nested folder structure with train/test/val splits and class subfolders
        """
        from PIL import Image
        
        print("Loading Chest X-Ray dataset...")
        images = []
        labels = []
        
        # Class map for chest x-ray (Normal vs Pneumonia)
        class_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        # Find the actual data path (handle nested structure)
        possible_paths = [
            os.path.join(data_path, 'chest_xray', 'chest_xray'),
            os.path.join(data_path, 'chest_xray'),
            data_path
        ]
        
        actual_path = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'train')):
                actual_path = path
                break
        
        if not actual_path:
            raise ValueError(f"Could not find chest_xray data structure in {data_path}")
        
        # Load from train/test/val folders
        for split_type in ['train', 'test', 'val']:
            split_path = os.path.join(actual_path, split_type)
            if not os.path.exists(split_path):
                continue
            
            for class_name, class_idx in class_map.items():
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize(self.image_size)
                            images.append(np.array(img))
                            labels.append(class_idx)
                        except Exception as e:
                            print(f"  Warning: Could not load {img_file}: {str(e)}")
        
        X = np.array(images)
        y = np.array(labels)
        
        if len(X) == 0:
            raise ValueError(f"No images found in {actual_path}")
        
        print(f"✓ Loaded {len(X)} samples from {len(np.unique(y))} classes")
        return X, y
    
    def preprocess_data(self, X):
        """
        Preprocess images: resize and normalize, handle grayscale -> RGB conversion
        
        Args:
            X: Input images (numpy array)
        
        Returns:
            Preprocessed images normalized to [0, 1] with 3 channels
        """
        from PIL import Image
        
        # Handle grayscale to RGB conversion (2D or 3D single channel)
        if X.ndim == 2:  # 2D grayscale: H x W
            X = np.expand_dims(X, axis=-1)  # H x W x 1
        elif X.ndim == 3 and X.shape[-1] == 1:  # Already 3D with 1 channel: N x H x W x 1
            pass  # Keep as is
        
        # Expand single channel to 3 channels if needed
        if X.ndim == 3 and X.shape[-1] == 1:
            X = np.repeat(X, 3, axis=-1)  # H x W x 1 -> H x W x 3
        elif X.ndim == 4 and X.shape[-1] == 1:
            X = np.repeat(X, 3, axis=-1)  # N x H x W x 1 -> N x H x W x 3
        
        # Resize if needed
        if X.ndim == 3 and X.shape[0:2] != self.image_size:
            resized_X = []
            for img in X:
                pil_img = Image.fromarray(img.astype('uint8')) if X.dtype == 'uint8' else Image.fromarray((img * 255).astype('uint8'))
                resized_img = pil_img.resize(self.image_size)
                resized_X.append(np.array(resized_img))
            X = np.array(resized_X)
        elif X.ndim == 4:  # Batch of images
            resized_X = []
            for img in X:
                pil_img = Image.fromarray(img.astype('uint8')) if X.dtype == 'uint8' else Image.fromarray((img * 255).astype('uint8'))
                resized_img = pil_img.resize(self.image_size)
                resized_X.append(np.array(resized_img))
            X = np.array(resized_X)
        
        # Normalize
        X = X.astype('float32') / 255.0
        
        print(f"✓ Preprocessed data shape: {X.shape}")
        return X
    
    def split_data(self, X, y):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Input data
            y: Labels
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = self.dataset_config['train_split']
        val_split = self.dataset_config['val_split']
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            train_size=train_split,
            random_state=42,
            stratify=y
        )
        
        # Second split: val vs test
        val_ratio = val_split / (val_split + self.dataset_config['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"✓ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_augmentation(self, X_train):
        """
        Apply data augmentation
        
        Args:
            X_train: Training images
        
        Returns:
            Transform object for PyTorch
        """
        augmentation_transform = transforms.Compose([
            transforms.RandomRotation(self.augmentation_config['rotation_range']),
            transforms.RandomAffine(0, translate=(
                self.augmentation_config['width_shift_range'],
                self.augmentation_config['height_shift_range']
            )),
            transforms.RandomHorizontalFlip(p=1.0 if self.augmentation_config['horizontal_flip'] else 0.0),
            transforms.RandomAffine(0, scale=(1 - self.augmentation_config['zoom_range'], 
                                             1 + self.augmentation_config['zoom_range'])),
        ])
        
        print("✓ Data augmentation configured")
        return augmentation_transform
    
    def load_and_prepare(self, dataset_name='Brain-MRI'):
        """
        Complete pipeline: load, preprocess, and split data
        
        Args:
            dataset_name: Name of dataset to load ('Brain-MRI' or 'Chest-XRay')
        
        Returns:
            Dictionary with train/val/test data and labels
        """
        # Load data
        if dataset_name == 'Brain-MRI':
            X, y = self.load_brain_mri()
        elif dataset_name == 'Chest-XRay':
            X, y = self.load_chest_xray()
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Use 'Brain-MRI' or 'Chest-XRay'")
        
        # Preprocess
        X = self.preprocess_data(X)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Rearrange for PyTorch (needs channels first: C x H x W)
        X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
        X_val_tensor = X_val_tensor.permute(0, 3, 1, 2)
        X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
        
        # Apply augmentation
        augmentation_transform = self.apply_augmentation(X_train_tensor)
        
        return {
            'X_train': X_train_tensor,
            'X_val': X_val_tensor,
            'X_test': X_test_tensor,
            'y_train': y_train_tensor,
            'y_val': y_val_tensor,
            'y_test': y_test_tensor,
            'augmentation': augmentation_transform,
            'num_classes': self.num_classes,
        }
