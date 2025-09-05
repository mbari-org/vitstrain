# vitstrain
# Filename: src/plot_utils.py
# Description: Utilities for plotting performance metrics
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, average_precision_score



def plot_multiclass_pr_curves(y_true, y_prob, class_names, model_name):
    """Plot precision-recall curves for multi-class classification."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    plt.figure(figsize=(15, 12))

    # Calculate precision-recall for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])

        plt.plot(recall, precision, label=f'{class_names[i]} (AP={avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Multi-class Precision-Recall Curves - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    d = f"{datetime.now():%Y-%m-%d_%H%M%S}"
    pr_curve_path = Path(model_name) / f"pr_curves_{model_name}_{d}.png"
    plt.savefig(pr_curve_path.as_posix(), dpi=300, bbox_inches='tight')
    plt.close()

    return pr_curve_path
