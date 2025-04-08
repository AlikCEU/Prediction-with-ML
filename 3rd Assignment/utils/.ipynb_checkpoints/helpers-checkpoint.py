# utils/helpers.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)

sns.set_palette("colorblind")
plt.style.use('seaborn-v0_8-whitegrid')


def plot_confusion(y_true, y_pred, labels, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_basic_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }


def expected_loss(y_true, y_pred, fp_cost=1, fn_cost=5):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * fp_cost + fn * fn_cost


def find_best_threshold(model, X, y, fp_cost=1, fn_cost=5, steps=100):
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, steps)
    best_t = 0.5
    min_cost = float('inf')

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = expected_loss(y, y_pred, fp_cost, fn_cost)
        if cost < min_cost:
            best_t = t
            min_cost = cost

    return best_t, min_cost


def plot_roc_curves(results_dict):
    plt.figure(figsize=(10, 8))
    for label, result in results_dict.items():
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_proba'])
        auc = roc_auc_score(result['y_true'], result['y_proba'])
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(results_dict):
    plt.figure(figsize=(10, 8))
    for label, result in results_dict.items():
        precision, recall, _ = precision_recall_curve(result['y_true'], result['y_proba'])
        plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()