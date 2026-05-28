"""
Sends identical patient profiles to the /predict API endpoint twice - once with
gender=1 (Male) and once with gender=2 (Female) - then measures the delta in
predicted cardiovascular risk. This isolates the model's learned gender signal
from population-level differences in the aggregate fairness analysis.
Usage:
    uvicorn api.main:app --port 8000
    python3 src/data/counterfactual_gender.py
"""

import argparse
import sys
import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PLOTS = PROJECT_ROOT / "output" / "plots"
OUTPUT_METRICS = PROJECT_ROOT / "output" / "metrics"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests


AGES = [35, 45, 55, 65]
HEIGHTS = [160, 175]
WEIGHTS = [60, 80, 100]
SMOKE_ALCO = [(0, 0), (1, 0), (0, 1)]
ACTIVE = [0, 1]
CHOLESTEROL = [1, 2, 3]


def build_patient_grid():
    """Cartesian product over clinically meaningful axes"""
    rows = []
    combos = itertools.product(AGES, HEIGHTS, WEIGHTS, SMOKE_ALCO, ACTIVE, CHOLESTEROL)
    for age, height, weight, (smoke, alco), active, chol in combos:
        rows.append({
            "age_years": age,
            "height": height,
            "weight": weight,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "cholesterol": chol,
        })
    return rows


def predict(url: str, patient: dict, gender: int) -> dict:
    payload = {**patient, "gender": gender}
    resp = requests.post(f"{url}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def run_probe(url: str, patients: list[dict]) -> pd.DataFrame:
    records = []
    n = len(patients)
    for i, patient in enumerate(patients, 1):
        print(f"\r  {i}/{n}", end="", flush=True)
        try:
            male_r = predict(url, patient, gender=1)
            female_r = predict(url, patient, gender=2)
        except requests.RequestException as e:
            print(f"\n  Error on patient {i}: {e}")
            continue

        male_pct = male_r["risk_percent"]
        female_pct = female_r["risk_percent"]
        delta = female_pct - male_pct

        bmi = patient["weight"] / (patient["height"] / 100) ** 2

        records.append({
            **patient,
            "bmi": round(bmi, 1),
            "male_risk": male_pct,
            "female_risk": female_pct,
            "delta": round(delta, 2),
            "abs_delta": round(abs(delta), 2),
        })
    print()
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    print("\nCounterfactual Gender Delta (Female risk - Male risk)\n")
    print(f"Patients probed       : {len(df)}")
    print(f"Delta range           : {df['delta'].min():+.1f}% to {df['delta'].max():+.1f}%")
    print(f"Mean delta            : {df['delta'].mean():+.2f}%")
    print(f"Median delta          : {df['delta'].median():+.2f}%")
    print(f"Std                   : {df['delta'].std():.2f}%")
    print(f"Females always higher : {(df['delta'] > 0).sum()} / {len(df)}")
    print(f"Males always higher   : {(df['delta'] < 0).sum()} / {len(df)}")
    print(f"No change             : {(df['delta'] == 0).sum()} / {len(df)}")

    print("\n--- Delta by Cholesterol level ---")
    print(df.groupby("cholesterol")["delta"].agg(["mean", "std", "min", "max"]).round(2).to_string())

    print("\n--- Delta by Age ---")
    print(df.groupby("age_years")["delta"].agg(["mean", "std", "min", "max"]).round(2).to_string())

    print("\n--- 10 largest absolute deltas ---")
    top = df.nlargest(10, "abs_delta")[
        ["age_years", "bmi", "cholesterol", "smoke", "alco", "active",
         "male_risk", "female_risk", "delta"]
    ]
    print(top.to_string(index=False))


def plot_results(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Histogram of deltas
    ax = axes[0, 0]
    ax.hist(df["delta"], bins=30, color="#7b68ee", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#c00", linewidth=1.2, linestyle="--", label="No difference")
    ax.axvline(df["delta"].mean(), color="#111", linewidth=1.2, linestyle="-",
               label=f"Mean {df['delta'].mean():+.1f}%")
    ax.set_xlabel("Delta (Female risk % - Male risk %)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Distribution of Gender Delta", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 2. Male vs Female scatter
    ax = axes[0, 1]
    sc = ax.scatter(df["male_risk"], df["female_risk"], c=df["delta"],
                    cmap="RdBu_r", alpha=0.4, s=18, vmin=-10, vmax=10)
    lim_min = min(df["male_risk"].min(), df["female_risk"].min()) - 1
    lim_max = max(df["male_risk"].max(), df["female_risk"].max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, label="No difference")
    plt.colorbar(sc, ax=ax, label="Delta (%)")
    ax.set_xlabel("Male risk %", fontsize=9)
    ax.set_ylabel("Female risk %", fontsize=9)
    ax.set_title("Male vs Female Predicted Risk", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # 3. Delta by age
    ax = axes[1, 0]
    age_groups = df.groupby("age_years")["delta"]
    positions = sorted(df["age_years"].unique())
    data_by_age = [age_groups.get_group(a).values for a in positions]
    bp = ax.boxplot(data_by_age, tick_labels=positions, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#b0c4de")
    ax.axhline(0, color="#c00", linewidth=1, linestyle="--")
    ax.set_xlabel("Age (years)", fontsize=9)
    ax.set_ylabel("Delta (%)", fontsize=9)
    ax.set_title("Gender Delta by Age", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 4. Delta by cholesterol + active
    ax = axes[1, 1]
    chol_labels = {1: "Normal", 2: "Above normal", 3: "Well above"}
    for active_val, color, marker in [(1, "#4a90d9", "o"), (0, "#e06090", "s")]:
        sub = df[df["active"] == active_val]
        means = sub.groupby("cholesterol")["delta"].mean()
        ax.plot(
            [chol_labels[c] for c in means.index],
            means.values,
            marker=marker, color=color, linewidth=1.5, markersize=7,
            label=f"Active={active_val}"
        )
    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cholesterol", fontsize=9)
    ax.set_ylabel("Mean delta (%)", fontsize=9)
    ax.set_title("Mean Delta by Cholesterol & Activity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Counterfactual Gender Fairness - Same Patient, Gender Flipped", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved: {out_path}")


def main():
    default_out = str(OUTPUT_PLOTS / "07_counterfactual_gender_after_fix.png")

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running API")
    parser.add_argument("--out", default=default_out,
                        help="Path for the output plot")
    args = parser.parse_args()

    try:
        requests.get(f"{args.url}/models", timeout=5).raise_for_status()
    except requests.RequestException:
        print(f"ERROR: API not reachable at {args.url}")
        print("Start it with:  uvicorn api.main:app --port 8000")
        sys.exit(1)

    patients = build_patient_grid()
    print(f"Probing {len(patients)} patient profiles (x2 for gender)...")
    df = run_probe(args.url, patients)

    if df.empty:
        print("No results - check the API logs.")
        sys.exit(1)

    print_summary(df)

    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    plot_results(df, args.out)

    csv_path = OUTPUT_METRICS / "counterfactual_gender.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved: {csv_path}")


if __name__ == "__main__":
    main()
