import numpy as np
import time
import torch
import torch.nn as nn

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torchvision.models import resnet50, efficientnet_b0
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class FeatureExtractor:

    def __init__(self, model_name='EfficientNet-B0', device='cpu'):
        self.model_name = model_name
        self.device = device

        self.feature_extractor = self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name}...")

        if self.model_name == 'ResNet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif self.model_name == 'EfficientNet-B0':
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            backbone = nn.Sequential(*list(model.children())[:-1])

        else:
            raise ValueError("Unsupported model")

        backbone = backbone.to(self.device)
        backbone.eval()

        for p in backbone.parameters():
            p.requires_grad = False

        return backbone

    def extract_features(self, images, batch_size=32):

        features = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):

                batch = images[i:i+batch_size].to(self.device)

                batch = self._normalize(batch)

                out = self.feature_extractor(batch)

                out = out.view(out.size(0), -1)

                features.append(out.cpu().numpy())

        return np.concatenate(features, axis=0)

    def _normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)
        return (x - mean) / std


class MLClassifierApproach1:

    def __init__(self, config):
        self.config = config
        self.model_config = config['MODEL_CONFIG']
        self.ml_config = config['ML_CLASSIFIER_CONFIG']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.scaler = StandardScaler()
        self.extractor = None
        self.classifier = None

    def train(self, X_train, y_train, X_val, y_val, augmentation_transform=None):

        start = time.time()

        self.extractor = FeatureExtractor(
            model_name=self.model_config['architecture'],
            device=self.device
        )

        X_train_feat = self.extractor.extract_features(X_train)
        X_val_feat = self.extractor.extract_features(X_val)

        # normalize
        X_train_feat = self.scaler.fit_transform(X_train_feat)
        X_val_feat = self.scaler.transform(X_val_feat)

        y_train = y_train.cpu().numpy()
        y_val = y_val.cpu().numpy()

        # classifier
        if self.ml_config['classifier_type'] == 'SVM':
            self.classifier = SVC(
                kernel=self.ml_config['svm_kernel'],
                C=self.ml_config['svm_C']
            )
        else:
            self.classifier = LogisticRegression(
                max_iter=self.ml_config['lr_max_iter']
            )

        self.classifier.fit(X_train_feat, y_train)

        train_time = time.time() - start

        val_pred = self.classifier.predict(X_val_feat)

        print("Validation Accuracy:", accuracy_score(y_val, val_pred))

        return {
            'training_time': train_time,
            'val_pred': val_pred
        }

    def evaluate(self, X_test, y_test):

        X_test_feat = self.extractor.extract_features(X_test)
        X_test_feat = self.scaler.transform(X_test_feat)

        y_test = y_test.cpu().numpy()

        pred = self.classifier.predict(X_test_feat)

        return {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, pred, average='weighted', zero_division=0),
            'y_pred': pred
        }