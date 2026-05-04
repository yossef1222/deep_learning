"""
Utility functions for the Deep Learning project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path


def create_directories(paths):
    """Create all required directories if they don't exist"""
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("✓ All directories created/verified")


def save_metrics(y_true, y_pred, approach_name, save_path):
    """
    Save evaluation metrics to a file
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        approach_name: Name of the approach (e.g., 'Approach-1', 'Approach-2')
        save_path: Path to save metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    with open(os.path.join(save_path, f'{approach_name}_metrics.txt'), 'w') as f:
        f.write(f"{'='*50}\n")
        f.write(f"{approach_name} - Evaluation Metrics\n")
        f.write(f"{'='*50}\n\n")
        f.write(classification_report(y_true, y_pred))
    
    return report


def plot_confusion_matrix(y_true, y_pred, approach_name, class_names, save_path):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        approach_name: Name of the approach
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{approach_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{approach_name}_confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"✓ Confusion matrix saved: {approach_name}")


def plot_training_history(history, approach_name, save_path):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history dictionary with keys ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        approach_name: Name of the approach
        save_path: Path to save the plot
    """
    if history is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'loss' in history and 'val_loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{approach_name} - Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Accuracy plot
    if 'accuracy' in history and 'val_accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title(f'{approach_name} - Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{approach_name}_training_history.png'), dpi=300)
    plt.close()
    print(f"✓ Training history plot saved: {approach_name}")


def plot_comparison(results, save_path):
    """
    Plot comparison between Approach-1 and Approach-2
    
    Args:
        results: Dictionary with results from both approaches
        save_path: Path to save the plot
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    approach1_scores = [results['approach1'].get(metric, 0) for metric in metrics_names]
    approach2_scores = [results['approach2'].get(metric, 0) for metric in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, approach1_scores, width, label='Approach-1 (DL+ML)', alpha=0.8)
    plt.bar(x + width/2, approach2_scores, width, label='Approach-2 (End-to-End)', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: DL+ML vs End-to-End DL')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'approach_comparison.png'), dpi=300)
    plt.close()
    print("✓ Comparison plot saved")


def print_summary(results):
    """Print a summary of results"""
    print("\n" + "="*60)
    print("PROJECT RESULTS SUMMARY")
    print("="*60)
    
    print("\nApproach-1 (DL-based Feature Extraction + ML Classifier):")
    print(f"  Accuracy:  {results['approach1'].get('accuracy', 'N/A'):.4f}")
    print(f"  Precision: {results['approach1'].get('precision', 'N/A'):.4f}")
    print(f"  Recall:    {results['approach1'].get('recall', 'N/A'):.4f}")
    print(f"  F1-Score:  {results['approach1'].get('f1_score', 'N/A'):.4f}")
    print(f"  Time:      {results['approach1'].get('training_time', 'N/A'):.2f} seconds")
    
    print("\nApproach-2 (End-to-End DL Model):")
    print(f"  Accuracy:  {results['approach2'].get('accuracy', 'N/A'):.4f}")
    print(f"  Precision: {results['approach2'].get('precision', 'N/A'):.4f}")
    print(f"  Recall:    {results['approach2'].get('recall', 'N/A'):.4f}")
    print(f"  F1-Score:  {results['approach2'].get('f1_score', 'N/A'):.4f}")
    print(f"  Time:      {results['approach2'].get('training_time', 'N/A'):.2f} seconds")
    
    print("\n" + "="*60)
