"""
Compare three training strategies for handling the 65/35 gender imbalance:
  - Baseline    : no resampling
  - Weights     : per-sample weights that equalise gender contribution
  - SMOTE       : oversample women in the training set
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    clean_data, split_and_scale,
    compute_gender_weights, smote_balance_gender,
)
from src.models.train import (
    train_logistic_regression, train_naive_bayes, train_decision_tree,
    train_linear_svm, train_random_forest, train_knn,
    train_xgboost, train_lightgbm,
)
from src.models.fairness import compare_gender_fairness, METRICS

OUTPUT_PLOTS = PROJECT_ROOT / "output" / "plots"
OUTPUT_METRICS = PROJECT_ROOT / "output" / "metrics"

MODELS_TO_TRAIN = {
    "Logistic Regression": train_logistic_regression,
    "Naive Bayes": train_naive_bayes,
    "Decision Tree": train_decision_tree,
    "Linear SVM": train_linear_svm,
    "Random Forest": train_random_forest,
    "KNN": train_knn,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
}

STRATEGIES = ["Baseline", "Weights", "SMOTE"]
STRATEGY_COLORS = {
    "Baseline": "#888888",
    "Weights": "#4a90d9",
    "SMOTE": "#e06090",
}


# Training

def train_strategy(X_train, y_train, strategy, gender_train):
    trained = {}
    for name, fn in MODELS_TO_TRAIN.items():
        if strategy == "Weights":
            weights = compute_gender_weights(gender_train)
            try:
                model = fn(X_train, y_train)
                model.fit(X_train, y_train, sample_weight=weights)
            except TypeError:
                model = fn(X_train, y_train)
        elif strategy == "SMOTE":
            X_bal, y_bal = smote_balance_gender(X_train, y_train, gender_train)
            model = fn(X_bal, y_bal)
        else:
            model = fn(X_train, y_train)
        trained[name] = model
    return trained


# Counterfactual delta (local, no API)

def counterfactual_delta(models_dict, X_test_scaled, gender_test, gender_col_idx):
    """
    For every test sample, flip the gender feature and measure the change
    in mean ensemble probability. Returns a Series of deltas (female - male).
    """
    gender_arr = np.asarray(gender_test)

    X_male = X_test_scaled.copy()
    X_female = X_test_scaled.copy()
    male_scaled_val = X_test_scaled[gender_arr == 1, gender_col_idx].mean()
    female_scaled_val = X_test_scaled[gender_arr == 2, gender_col_idx].mean()

    X_male[:, gender_col_idx] = male_scaled_val
    X_female[:, gender_col_idx] = female_scaled_val

    probs_male = []
    probs_female = []

    for model in models_dict.values():
        if hasattr(model, "predict_proba"):
            probs_male.append(model.predict_proba(X_male)[:, 1])
            probs_female.append(model.predict_proba(X_female)[:, 1])
        else:
            # decision_function -> sigmoid for SVM
            from scipy.special import expit
            probs_male.append(expit(model.decision_function(X_male)))
            probs_female.append(expit(model.decision_function(X_female)))

    mean_male = np.mean(probs_male, axis=0) * 100
    mean_female = np.mean(probs_female, axis=0) * 100
    return mean_female - mean_male


# Plotting

def plot_comparison(fairness_all, delta_all, save_path):
    n_strategies = len(STRATEGIES)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # aggregate metric gap (Women - Men) per strategy
    ax = axes[0, 0]
    metric_labels = [m.capitalize() for m in METRICS]
    x = np.arange(len(metric_labels))
    bar_width = 0.25
    offsets = np.linspace(-(n_strategies - 1) / 2, (n_strategies - 1) / 2, n_strategies) * bar_width

    for i, strategy in enumerate(STRATEGIES):
        gaps = []
        for metric in METRICS:
            avg_men = np.mean([fairness_all[strategy][m]["Men"][metric] for m in MODELS_TO_TRAIN])
            avg_women = np.mean([fairness_all[strategy][m]["Women"][metric] for m in MODELS_TO_TRAIN])
            gaps.append(avg_women - avg_men)
        ax.bar(x + offsets[i], gaps, bar_width, label=strategy, color=STRATEGY_COLORS[strategy])

    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Gap (Women - Men)", fontsize=9)
    ax.set_title("Avg metric gap across all models", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # mean absolute counterfactual delta per strategy
    ax = axes[0, 1]
    mean_abs = [delta_all[s].abs().mean() for s in STRATEGIES]
    colors = [STRATEGY_COLORS[s] for s in STRATEGIES]
    bars = ax.bar(STRATEGIES, mean_abs, color=colors, width=0.4)
    for bar, val in zip(bars, mean_abs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean |delta| (%)", fontsize=9)
    ax.set_title("Mean absolute counterfactual gender delta", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # delta distribution per strategy
    ax = axes[1, 0]
    for strategy in STRATEGIES:
        ax.hist(delta_all[strategy], bins=40, alpha=0.5,
                label=strategy, color=STRATEGY_COLORS[strategy])
    ax.axvline(0, color="#c00", linewidth=1, linestyle="--")
    ax.set_xlabel("Delta (Female risk % - Male risk %)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Counterfactual delta distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # per-model recall by gender for each strategy
    ax = axes[1, 1]
    model_names = list(MODELS_TO_TRAIN.keys())
    x = np.arange(len(model_names))
    bar_width = 0.25
    offsets = np.linspace(-(n_strategies - 1) / 2, (n_strategies - 1) / 2, n_strategies) * bar_width

    for i, strategy in enumerate(STRATEGIES):
        recall_gaps = [
            fairness_all[strategy][m]["Women"]["recall"] - fairness_all[strategy][m]["Men"]["recall"]
            for m in model_names
        ]
        ax.bar(x + offsets[i], recall_gaps, bar_width, label=strategy, color=STRATEGY_COLORS[strategy])

    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Recall gap (Women - Men)", fontsize=9)
    ax.set_title("Recall gap per model", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Gender Balance Strategy Comparison", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved: {save_path}")


def print_summary(fairness_all, delta_all):
    print("\nOverall accuracy per gender (averaged across models)\n")
    header = f"{'Strategy':<12} {'Men acc':>9} {'Women acc':>10} {'Gap':>7} | {'|delta| mean':>12} {'|delta| std':>11}"
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        men_acc = np.mean([fairness_all[s][m]["Men"]["accuracy"] for m in MODELS_TO_TRAIN])
        women_acc = np.mean([fairness_all[s][m]["Women"]["accuracy"] for m in MODELS_TO_TRAIN])
        gap = women_acc - men_acc
        d = delta_all[s].abs()
        print(f"{s:<12} {men_acc:>9.3f} {women_acc:>10.3f} {gap:>+7.3f} | {d.mean():>12.2f}% {d.std():>10.2f}%")

    print("\nOverall metric gaps (Women - Men, averaged across models)\n")
    print(f"{'Strategy':<12}", end="")
    for m in METRICS:
        print(f"  {m.capitalize():>10}", end="")
    print()
    print("-" * (12 + 12 * len(METRICS)))
    for s in STRATEGIES:
        print(f"{s:<12}", end="")
        for metric in METRICS:
            gap = np.mean([
                fairness_all[s][m]["Women"][metric] - fairness_all[s][m]["Men"][metric]
                for m in MODELS_TO_TRAIN
            ])
            print(f"  {gap:>+10.3f}", end="")
        print()


# Main

def main():
    print("Loading data...")
    df_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "cardio_train.csv", sep=";")
    df_cleaned = clean_data(df_raw)

    print("Splitting and scaling (feature-engineered)...")
    X_train, X_test, y_train, y_test, _, X_train_df = split_and_scale(
        df_cleaned, use_feature_engineering=True
    )

    gender_train = df_cleaned.loc[X_train_df.index, "gender"]
    gender_test = df_cleaned.loc[y_test.index, "gender"]

    # gender is the first column in the engineered feature set
    feature_cols = list(X_train_df.columns)
    gender_col_idx = feature_cols.index("gender")

    fairness_all = {}
    delta_all = {}

    for strategy in STRATEGIES:
        print(f"\nTraining [{strategy}]...")
        models = train_strategy(X_train, y_train, strategy, gender_train)

        print(f"  Evaluating fairness...")
        fairness_all[strategy] = compare_gender_fairness(models, X_test, y_test, gender_test)

        print(f"  Computing counterfactual deltas...")
        deltas = counterfactual_delta(models, X_test, gender_test, gender_col_idx)
        delta_all[strategy] = pd.Series(deltas)

    print_summary(fairness_all, delta_all)

    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)

    plot_path = OUTPUT_PLOTS / "08_gender_balance_comparison.png"
    plot_comparison(fairness_all, delta_all, plot_path)

    # Save raw delta data
    delta_df = pd.DataFrame({s: delta_all[s].values for s in STRATEGIES})
    delta_df.to_csv(OUTPUT_METRICS / "gender_balance_deltas.csv", index=False)
    print(f"Raw deltas saved: {OUTPUT_METRICS / 'gender_balance_deltas.csv'}")


if __name__ == "__main__":
    main()
