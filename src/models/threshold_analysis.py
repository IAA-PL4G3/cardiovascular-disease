import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score
)


def apply_threshold(probabilities, threshold=0.5):
    # convert probabilities to binary predictions based on threshold
    return (probabilities >= threshold).astype(int)


def calculate_confusion_metrics(y_true, y_pred):
    # calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    return metrics


def compare_thresholds(probabilities, y_test, thresholds=[0.5, 0.6], model_name="Model"):
    # compare predictions using different thresholds and calculate metrics for each
    results = {}
    
    for threshold in thresholds:
        predictions = apply_threshold(probabilities, threshold)
        metrics = calculate_confusion_metrics(y_test, predictions)
        results[f"Threshold: {threshold:.0%}"] = metrics
    
    df = pd.DataFrame(results).T
    
    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    df_metrics = df[metric_cols]
    print(f"\n{model_name} - Threshold Comparison:")
    print(df_metrics.to_string())
    
    return df


def print_confusion_breakdown(metrics, model_name="Model", threshold=0.5):
    # print breakdown of confusion matrix components (TP, FP, TN, FN) for a given threshold
    tp = metrics["TP"]
    fp = metrics["FP"]
    tn = metrics["TN"]
    fn = metrics["FN"]
    
    total = tp + fp + tn + fn
    
    print(f"\n{model_name} (Threshold: {threshold:.0%}) - Confusion Matrix Breakdown:")
    print(f"  True Positives (TP):  {tp:5d} ({tp/total*100:5.2f}%)")
    print(f"  False Positives (FP): {fp:5d} ({fp/total*100:5.2f}%)")
    print(f"  True Negatives (TN):  {tn:5d} ({tn/total*100:5.2f}%)")
    print(f"  False Negatives (FN): {fn:5d} ({fn/total*100:5.2f}%)")
    print(f"  Total: {total}")
