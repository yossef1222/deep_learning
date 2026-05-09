"""
Plot Confusion Matrix for Chest X-Ray Dataset (Load Pre-trained Models)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import joblib
from config import AUGMENTATION_CONFIG, ML_CLASSIFIER_CONFIG, TRAINING_CONFIG, PATHS
from data_loader import DataLoader
from approach1_ml_classifier import MLClassifierApproach1
from approach2_end_to_end import EndToEndDLModel


def main():
    print("\n" + "=" * 70)
    print("CHEST X-RAY CONFUSION MATRIX VISUALIZATION (LOADED MODELS)")
    print("=" * 70)
    
    os.makedirs(PATHS['results'], exist_ok=True)
    os.makedirs(PATHS['plots'], exist_ok=True)
    os.makedirs(PATHS['metrics'], exist_ok=True)
    
    # ==================== Dataset Configuration ====================
    dataset_config = {
        'name': 'Chest-XRay',
        'data_dir': './data/raw',
        'image_size': (224, 224),
        'num_classes': 2,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
    }
    
    model_configs = [
        {'architecture': 'ResNet50', 'pretrained': True},
        {'architecture': 'EfficientNet-B0', 'pretrained': True}
    ]
    
    base_config = {
        'DATASET_CONFIG': dataset_config,
        'TRAINING_CONFIG': TRAINING_CONFIG,
        'AUGMENTATION_CONFIG': AUGMENTATION_CONFIG,
        'ML_CLASSIFIER_CONFIG': ML_CLASSIFIER_CONFIG
    }
    
    print("\nLoading Chest X-Ray dataset...")
    print(f"Data directory: ./data/raw/Chest_X-Ray")
    
    # Check if data exists
    chest_xray_path = './data/raw/Chest_X-Ray'
    if not os.path.exists(chest_xray_path):
        print(f"✗ Data path does not exist: {chest_xray_path}")
        return
    
    print("✓ Data path exists")
    
    data_loader = DataLoader(base_config)
    print("Creating data loader...")
    
    data = data_loader.load_and_prepare(dataset_name='Chest-XRay')
    print("✓ Data loaded successfully")
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    augmentation_transform = data['augmentation']
    
    class_names = ['NORMAL', 'PNEUMONIA']
    
    for model_config in model_configs:
        print(f"\n{'='*70}")
        print(f"Processing {model_config['architecture']}")
        print(f"{'='*70}")
        
        config = base_config.copy()
        config['MODEL_CONFIG'] = model_config
        
        exp_name = f"Chest-XRay_{model_config['architecture']}"
        model_dir = os.path.join(PATHS['models'], exp_name)
        
        # ==================== APPROACH 1 (LOAD) ====================
        print(f"\nApproach-1: Feature Extraction + ML Classifier (Loading...)")
        print("-" * 60)
        
        approach1 = MLClassifierApproach1(config)
        
        # Load pre-trained models
        feature_extractor_path = os.path.join(model_dir, "feature_extractor.pth")
        ml_model_path = os.path.join(model_dir, "ml_model.pkl")
        
        if os.path.exists(feature_extractor_path) and os.path.exists(ml_model_path):
            # Load feature extractor state dict
            approach1.feature_extractor.load_state_dict(
                torch.load(feature_extractor_path, map_location=approach1.device)
            )
            # Load ML classifier
            approach1.classifier = joblib.load(ml_model_path)
            print(f"✓ Loaded feature extractor from: {feature_extractor_path}")
            print(f"✓ Loaded ML classifier from: {ml_model_path}")
        else:
            print(f"✗ Model files not found. Training instead...")
            approach1.train(X_train, y_train, X_val, y_val, augmentation_transform)
            torch.save(
                approach1.feature_extractor.state_dict(),
                feature_extractor_path
            )
            joblib.dump(approach1.classifier, ml_model_path)
        
        a1_result = approach1.evaluate(X_test, y_test)
        y_pred_a1 = a1_result['y_pred']
        
        # Confusion Matrix for Approach-1
        cm_a1 = confusion_matrix(y_test.cpu().numpy(), y_pred_a1)
        
        print(f"Accuracy: {a1_result['accuracy']:.4f}")
        print(f"Confusion Matrix:\n{cm_a1}")
        
        # ==================== APPROACH 2 (LOAD) ====================
        print(f"\nApproach-2: End-to-End Deep Learning (Loading...)")
        print("-" * 60)
        
        approach2 = EndToEndDLModel(config)
        approach2.build_model()
        
        # Load pre-trained model
        model_path = os.path.join(model_dir, "end_to_end_model.pth")
        
        if os.path.exists(model_path):
            approach2.model.load_state_dict(
                torch.load(model_path, map_location=approach2.device)
            )
            print(f"✓ Loaded model from: {model_path}")
        else:
            print(f"✗ Model file not found. Training instead...")
            approach2.train(X_train, y_train, X_val, y_val, augmentation_transform)
            torch.save(approach2.model.state_dict(), model_path)
        
        a2_result = approach2.evaluate(X_test, y_test)
        y_pred_a2 = a2_result['y_pred']
        
        # Confusion Matrix for Approach-2
        cm_a2 = confusion_matrix(y_test.cpu().numpy(), y_pred_a2)
        
        print(f"Accuracy: {a2_result['accuracy']:.4f}")
        print(f"Confusion Matrix:\n{cm_a2}")
        
        # ==================== Plot Confusion Matrices ====================
        print(f"\nGenerating visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Approach-1 Confusion Matrix
        sns.heatmap(
            cm_a1,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            annot_kws={'size': 12}
        )
        axes[0].set_title(f'Approach-1: Feature Extraction + ML\n(Accuracy: {a1_result["accuracy"]:.4f})', 
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)
        
        # Approach-2 Confusion Matrix
        sns.heatmap(
            cm_a2,
            annot=True,
            fmt='d',
            cmap='Greens',
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            annot_kws={'size': 12}
        )
        axes[1].set_title(f'Approach-2: End-to-End DL\n(Accuracy: {a2_result["accuracy"]:.4f})', 
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        
        plt.suptitle(f'Chest X-Ray Dataset - {model_config["architecture"]} Model', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(PATHS['plots'], f'ChestXRay_{model_config["architecture"]}_confusion_matrices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
        
        # ==================== Create Individual Plots ====================
        # Approach-1 Individual Plot
        fig_a1, ax_a1 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_a1,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax_a1,
            annot_kws={'size': 14},
            square=True
        )
        ax_a1.set_title(f'Chest X-Ray - {model_config["architecture"]}\nApproach-1: Feature Extraction + ML\nAccuracy: {a1_result["accuracy"]:.4f}', 
                       fontsize=12, fontweight='bold')
        ax_a1.set_ylabel('True Label', fontsize=11)
        ax_a1.set_xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()
        
        plot_path_a1 = os.path.join(PATHS['plots'], f'ChestXRay_{model_config["architecture"]}_A1_confusion_matrix.png')
        plt.savefig(plot_path_a1, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path_a1}")
        plt.close()
        
        # Approach-2 Individual Plot
        fig_a2, ax_a2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_a2,
            annot=True,
            fmt='d',
            cmap='Greens',
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax_a2,
            annot_kws={'size': 14},
            square=True
        )
        ax_a2.set_title(f'Chest X-Ray - {model_config["architecture"]}\nApproach-2: End-to-End DL\nAccuracy: {a2_result["accuracy"]:.4f}', 
                       fontsize=12, fontweight='bold')
        ax_a2.set_ylabel('True Label', fontsize=11)
        ax_a2.set_xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()
        
        plot_path_a2 = os.path.join(PATHS['plots'], f'ChestXRay_{model_config["architecture"]}_A2_confusion_matrix.png')
        plt.savefig(plot_path_a2, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path_a2}")
        plt.close()
        
        # ==================== Save Metrics ====================
        metrics_content = f"""
CHEST X-RAY DATASET - {model_config['architecture']} MODEL
{'='*70}

APPROACH-1: Feature Extraction + ML Classifier
{'-'*70}
Accuracy:  {a1_result['accuracy']:.4f}
Precision: {a1_result['precision']:.4f}
Recall:    {a1_result['recall']:.4f}
F1-Score:  {a1_result['f1_score']:.4f}

Confusion Matrix:
{cm_a1}

Class Distribution:
  NORMAL:    {np.sum(cm_a1[0])} (TP: {cm_a1[0, 0]}, FP: {cm_a1[1, 0]})
  PNEUMONIA: {np.sum(cm_a1[1])} (TP: {cm_a1[1, 1]}, FP: {cm_a1[0, 1]})

APPROACH-2: End-to-End Deep Learning
{'-'*70}
Accuracy:  {a2_result['accuracy']:.4f}
Precision: {a2_result['precision']:.4f}
Recall:    {a2_result['recall']:.4f}
F1-Score:  {a2_result['f1_score']:.4f}

Confusion Matrix:
{cm_a2}

Class Distribution:
  NORMAL:    {np.sum(cm_a2[0])} (TP: {cm_a2[0, 0]}, FP: {cm_a2[1, 0]})
  PNEUMONIA: {np.sum(cm_a2[1])} (TP: {cm_a2[1, 1]}, FP: {cm_a2[0, 1]})

COMPARISON
{'-'*70}
Accuracy Difference (A2 - A1): {a2_result['accuracy'] - a1_result['accuracy']:+.4f}
Precision Difference (A2 - A1): {a2_result['precision'] - a1_result['precision']:+.4f}
Recall Difference (A2 - A1): {a2_result['recall'] - a1_result['recall']:+.4f}
F1-Score Difference (A2 - A1): {a2_result['f1_score'] - a1_result['f1_score']:+.4f}
"""
        
        metrics_path = os.path.join(PATHS['metrics'], f'ChestXRay_{model_config["architecture"]}_confusion_matrices.txt')
        with open(metrics_path, 'w') as f:
            f.write(metrics_content)
        print(f"✓ Saved metrics: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✓ CHEST X-RAY CONFUSION MATRIX VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
