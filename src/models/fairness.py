import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


GENDER_LABELS = {1: "Men", 2: "Women"}
GENDER_COLORS = {"Men": "#4a90d9", "Women": "#e06090"}
METRICS = ["accuracy", "precision", "recall", "f1"]


def evaluate_by_gender(model, X_test_scaled, y_test, gender_series):
    """
    Evaluate a model's classification metrics split by gender
    """
    results = {}
    gender_arr = gender_series.values if hasattr(gender_series, "values") else np.array(gender_series)
    y_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    for gender_val, label in GENDER_LABELS.items():
        mask = gender_arr == gender_val
        X_g = X_test_scaled[mask]
        y_g = y_arr[mask]

        y_pred = model.predict(X_g)

        results[label] = {
            "n": int(mask.sum()),
            "accuracy": accuracy_score(y_g, y_pred),
            "precision": precision_score(y_g, y_pred, zero_division=0),
            "recall": recall_score(y_g, y_pred, zero_division=0),
            "f1": f1_score(y_g, y_pred, zero_division=0),
        }

    return results


def compare_gender_fairness(models_dict, X_test_scaled, y_test, gender_series):
    """
    Evaluate all models by gender and return a dictionary of results for each model
    """
    return {
        name: evaluate_by_gender(model, X_test_scaled, y_test, gender_series)
        for name, model in models_dict.items()
    }


def print_gender_summary(all_results):
    """Print a formatted table of per-gender metrics for each model."""
    rows = []
    for model_name, gender_results in all_results.items():
        for gender_label, metrics in gender_results.items():
            rows.append({
                "Model": model_name,
                "Group": gender_label,
                "N": metrics["n"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            })

    df = pd.DataFrame(rows)
    df_fmt = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}")
    print(df_fmt.to_string(index=False))

    print("\nGap (Women - Men):")
    gap_rows = []
    for model_name, gender_results in all_results.items():
        if "Men" in gender_results and "Women" in gender_results:
            row = {"Model": model_name}
            for metric in METRICS:
                row[metric.capitalize()] = gender_results["Women"][metric] - gender_results["Men"][metric]
            gap_rows.append(row)
    gap_df = pd.DataFrame(gap_rows)
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        gap_df[col] = gap_df[col].map(lambda x: f"{x:+.3f}")
    print(gap_df.to_string(index=False))

    return df


def plot_gender_metrics(all_results, save_path=None):
    """
    4-subplot bar chart: one panel per metric (accuracy, precision, recall, f1).
    Each panel shows Men vs Women bars for all models.
    """
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    bar_width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        men_vals = [all_results[m]["Men"][metric] for m in model_names]
        women_vals = [all_results[m]["Women"][metric] for m in model_names]

        bars_m = ax.bar(x - bar_width / 2, men_vals, bar_width, label="Men", color=GENDER_COLORS["Men"])
        bars_w = ax.bar(x + bar_width / 2, women_vals, bar_width, label="Women", color=GENDER_COLORS["Women"])

        for bar in bars_m:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)
        for bar in bars_w:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Performance by Gender (Feature-Engineered)", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
