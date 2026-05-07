"""
Selective prediction (a.k.a. prediction with abstention / classification with
a reject option). The classifier outputs a probability; we keep predictions
only where the model's confidence — defined as max(p, 1 - p) — exceeds a
threshold, and abstain ("refer to a clinician") on the rest. This trades
coverage for accuracy and is a standard technique in clinical decision
support, where it is often more useful to be highly accurate on a confident
subset than to commit on every patient.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def confidence_from_proba(proba_pos: np.ndarray) -> np.ndarray:
    """Confidence of a binary classifier, i.e. distance of p from 0.5."""
    proba_pos = np.asarray(proba_pos)
    return np.maximum(proba_pos, 1.0 - proba_pos)


def selective_accuracy_curve(proba_pos, y_true,
                              coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                                         0.4, 0.3, 0.2, 0.1)) -> pd.DataFrame:
    """
    For each target coverage, keep the most-confident `coverage * N` predictions
    and report accuracy on that retained subset. Returns a DataFrame.
    """
    proba_pos = np.asarray(proba_pos)
    y_true    = np.asarray(y_true)
    pred      = (proba_pos >= 0.5).astype(int)
    conf      = confidence_from_proba(proba_pos)

    rows = []
    for cov in coverages:
        thr = float(np.quantile(conf, 1.0 - cov)) if cov < 1.0 else 0.5
        mask = conf >= thr
        n_kept = int(mask.sum())
        acc = accuracy_score(y_true[mask], pred[mask]) if n_kept else float('nan')
        rows.append({
            'target_coverage': cov,
            'confidence_threshold': round(thr, 4),
            'n_kept': n_kept,
            'actual_coverage': n_kept / len(y_true),
            'accuracy': acc,
        })
    return pd.DataFrame(rows)


def coverage_for_target_accuracy(proba_pos, y_true, target_acc: float = 0.85):
    """
    Find the maximum coverage at which selective-prediction accuracy reaches
    `target_acc`. Returns (coverage, threshold, accuracy, n_kept) or None
    if the target is unreachable.
    """
    proba_pos = np.asarray(proba_pos)
    y_true    = np.asarray(y_true)
    pred      = (proba_pos >= 0.5).astype(int)
    conf      = confidence_from_proba(proba_pos)

    order = np.argsort(-conf)  # most confident first
    correct = (pred == y_true)[order]
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    # find the largest k such that cumulative accuracy >= target
    eligible = np.where(cum_acc >= target_acc)[0]
    if len(eligible) == 0:
        return None
    k = int(eligible[-1]) + 1
    thr = float(conf[order][k - 1])
    return {
        'coverage': k / len(y_true),
        'threshold': thr,
        'accuracy': float(cum_acc[k - 1]),
        'n_kept': k,
    }


def subgroup_accuracy(df: pd.DataFrame, y_true, y_pred,
                      group_col: str, min_n: int = 50) -> pd.DataFrame:
    """Accuracy broken down by the values of `group_col` in `df`."""
    df = df.copy()
    df['_y_true'] = np.asarray(y_true)
    df['_y_pred'] = np.asarray(y_pred)
    rows = []
    for g, sub in df.groupby(group_col):
        if len(sub) < min_n:
            continue
        rows.append({
            group_col: g,
            'n': len(sub),
            'accuracy': accuracy_score(sub['_y_true'], sub['_y_pred']),
        })
    return pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
