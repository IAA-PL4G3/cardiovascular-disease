from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
import sys
import json
import os

def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model

def train_naive_bayes(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train, max_depth=7):
    # max_depth added to prevent severe overfitting
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    return model

def train_linear_svm(x_train, y_train):
    model = LinearSVC(max_iter=10000, random_state=42)
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train, n_estimators=140):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(x_train, y_train)
    return model

def train_knn(x_train, y_train, n_neighbors=5):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    return {
        "accuracy": acc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "predictions": predictions
    }

def train_all_models(x_train, y_train, x_test, y_test, feature_engineering_enabled=False):
    """
    Train all models and evaluate them
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        feature_engineering_enabled: Whether feature engineering was used
    
    Returns:
        Dictionary with models and metrics
    """
    print("Training all models...\n")
    models = {
        "Logistic Regression": train_logistic_regression(x_train, y_train),
        "Naive Bayes": train_naive_bayes(x_train, y_train),
        "Decision Tree": train_decision_tree(x_train, y_train, max_depth=7),
        "Linear SVM": train_linear_svm(x_train, y_train),
        "Random Forest": train_random_forest(x_train, y_train, n_estimators=140),
        "KNN": train_knn(x_train, y_train, n_neighbors=5)
    }
    
    results = {}
    print("Model Performance:\n")
    for model_name, model in models.items():
        metrics = evaluate_model(model, x_test, y_test)
        results[model_name] = metrics
        
        print(f"{model_name}:")
        for metric, value in metrics.items():
            if metric != "predictions":
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Save model with appropriate suffix
        suffix = "_with_feature_engineering" if feature_engineering_enabled else "_baseline"
        models_dir = os.path.join(os.path.dirname(__file__), "../../models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_').lower()}{suffix}.pkl")
        joblib.dump(model, model_path)
        print(f"  Saved to: {model_path}\n")
    
    return models, results

def compare_models_performance(results_baseline, results_engineered):
    """Compare performance between baseline and feature-engineered models"""
    
    print("COMPARISON: Baseline vs Feature-Engineered Models")
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score"]
    
    for model_name in results_baseline.keys():
        print(f"{model_name}:")
        baseline_metrics = results_baseline[model_name]
        engineered_metrics = results_engineered[model_name]
        
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            engineered_val = engineered_metrics[metric]
            diff = engineered_val - baseline_val
            diff_str = f"{diff:+.4f}" if diff else "0.0000"
            
            print(f"  {metric.capitalize()}:")
            print(f"    Baseline:  {baseline_val:.4f}")
            print(f"    Engineered: {engineered_val:.4f}")
            print(f"    Difference: {diff_str}")
        print()
        
if __name__ == "__main__":
    sys.path.insert(0, "../../")
    from src.features.build_features import clean_data, split_and_scale
    
    df = pd.read_csv("../../data/raw/cardio_train.csv", sep=";")
    df_cleaned = clean_data(df)
    
    print("TRAINING BASELINE MODELS (without feature engineering)")
    x_train_b, x_test_b, y_train_b, y_test_b, scaler_b = split_and_scale(
        df_cleaned, use_feature_engineering=False
    )
    models_baseline, results_baseline = train_all_models(
        x_train_b, y_train_b, x_test_b, y_test_b, feature_engineering_enabled=False
    )
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler_b, os.path.join(models_dir, "scaler_baseline.pkl"))
    
    print("TRAINING FEATURE-ENGINEERED MODELS")
    x_train_e, x_test_e, y_train_e, y_test_e, scaler_e = split_and_scale(
        df_cleaned, use_feature_engineering=True
    )
    models_engineered, results_engineered = train_all_models(
        x_train_e, y_train_e, x_test_e, y_test_e, feature_engineering_enabled=True
    )
    joblib.dump(scaler_e, os.path.join(models_dir, "scaler_with_feature_engineering.pkl"))
    
    # Compare results
    compare_models_performance(results_baseline, results_engineered)

