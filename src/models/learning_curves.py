"""
Learning Curves Analysis Module

This module provides functionality to generate and visualize learning curves
for analyzing bias-variance trade-offs in machine learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(model, X, y, cv=5, title="Learning Curve", train_sizes=None, n_jobs=-1):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Generate learning curve data
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, 
        cv=cv, 
        scoring="accuracy", 
        train_sizes=train_sizes, 
        n_jobs=n_jobs
    )
    
    # Compute mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot training curve with confidence band
    plt.plot(train_sizes_abs, train_scores_mean, marker='o', linewidth=2,
             label='Training Accuracy', color='#2E86AB')
    plt.fill_between(train_sizes_abs, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.2, color='#2E86AB')
    
    # Plot validation curve with confidence band
    plt.plot(train_sizes_abs, val_scores_mean, marker='s', linewidth=2,
             label='Validation Accuracy', color='#A23B72')
    plt.fill_between(train_sizes_abs, 
                     val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, 
                     alpha=0.2, color='#A23B72')
    
    # Formatting
    plt.xlabel('Training Set Size (samples)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {
        'train_sizes_abs': train_sizes_abs,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'val_scores_mean': val_scores_mean,
        'val_scores_std': val_scores_std
    }