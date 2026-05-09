
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models import resnet50, efficientnet_b0
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from tqdm import tqdm

class EndToEndDLModel:
    
    def __init__(self, config):

        self.config = config
        self.model_config = config['MODEL_CONFIG']
        self.training_config = config['TRAINING_CONFIG']
        self.dataset_config = config['DATASET_CONFIG']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = None
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    def build_model(self):
        """Build the end-to-end classification model"""
        print(f"Building {self.model_config['architecture']} for end-to-end classification...")
        
        # Load pre-trained model
        if self.model_config['architecture'] == 'ResNet50':
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif self.model_config['architecture'] == 'EfficientNet-B0':
            base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Architecture {self.model_config['architecture']} not supported. Use 'ResNet50' or 'EfficientNet-B0'")
        
        # Build full model
        class ClassificationModel(nn.Module):
            def __init__(self, base_model, num_features, num_classes):
                super(ClassificationModel, self).__init__()
                self.base_model = base_model
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(num_features, 256)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(256, num_classes)
            
            def forward(self, x):
                x = self.base_model(x)
                if x.dim() == 4:  # If features are still 4D from conv layers
                    x = self.global_pool(x)
                    x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        self.model = ClassificationModel(base_model, num_features, self.dataset_config['num_classes'])
        self.model = self.model.to(self.device)
        
        print(f"✓ Model built successfully")
        print(self.model)
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, augmentation_transform=None):
  
        if self.model is None:
            self.build_model()
        
        print("Training end-to-end model...")
        start_time = time.time()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'], 
                                 shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'])
        
        # Optimizer and loss
        if self.training_config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.training_config['learning_rate'])
        elif self.training_config['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), 
                                 lr=self.training_config['learning_rate'])
        else:  # rmsprop
            optimizer = optim.RMSprop(self.model.parameters(), 
                                     lr=self.training_config['learning_rate'])
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_count = 0
        
        for epoch in range(self.training_config['epochs']):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.training_config['epochs']}"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if augmentation_transform is not None:
                    batch_X = augmentation_transform(batch_X)
                
                # Normalize
                batch_X = self._normalize_batch(batch_X)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Clear cache to free memory
                torch.cuda.empty_cache()
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Normalize
                    batch_X = self._normalize_batch(batch_X)
                    
                    outputs = self.model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Store history
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            if self.training_config['early_stopping']:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_model = self.model.state_dict().copy()
                else:
                    patience_count += 1
                    if patience_count >= self.training_config['early_stopping_patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_model)
                        break
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'history': self.history,
        }
    
    def fine_tune(self, X_train, y_train, X_val, y_val):
   
        print("Fine-tuning model (unfreezing base model layers)...")
        
        # Unfreeze base model
        for param in self.model.base_model.parameters():
            param.requires_grad = True
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'], 
                                 shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'])
        
        # Use lower learning rate for fine-tuning
        lower_lr = self.training_config['learning_rate'] / 10
        if self.training_config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lower_lr)
        elif self.training_config['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lower_lr)
        else:
            optimizer = optim.RMSprop(self.model.parameters(), lr=lower_lr)
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Fine-tuning loop (fewer epochs)
        best_val_loss = float('inf')
        patience_count = 0
        finetune_epochs = self.training_config['epochs'] // 2
        
        start_time = time.time()
        
        for epoch in range(finetune_epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Fine-tune {epoch+1}/{finetune_epochs}"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Normalize
                batch_X = self._normalize_batch(batch_X)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Normalize
                    batch_X = self._normalize_batch(batch_X)
                    
                    outputs = self.model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.training_config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        tuning_time = time.time() - start_time
        print(f"✓ Fine-tuning completed in {tuning_time:.2f} seconds")
        
        return {
            'tuning_time': tuning_time,
        }
    
    def evaluate(self, X_test, y_test):
   
      
        print("Evaluating end-to-end model on test set...")
        
        self.model.eval()
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.training_config['batch_size'])
        
        y_pred_list = []
        y_test_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Evaluating"):
                batch_X = batch_X.to(self.device)
                
                # Normalize
                batch_X = self._normalize_batch(batch_X)
                
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                y_pred_list.append(predicted.cpu().numpy())
                y_test_list.append(batch_y.cpu().numpy())
        
        y_pred = np.concatenate(y_pred_list)
        y_test_np = np.concatenate(y_test_list)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_np, y_pred)
        precision = precision_score(y_test_np, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_np, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_np, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test_np, y_pred)
        
        print(f"✓ Approach-2 Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
        }
    
    def _normalize_batch(self, batch):
        """Normalize batch using ImageNet statistics"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (batch - mean) / std
