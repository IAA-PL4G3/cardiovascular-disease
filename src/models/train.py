from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from src.features.build_features import clean_data, split_and_scale
import pandas as pd
import joblib
import sys

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

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    return {
        "accuracy": acc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
    }


if __name__ == "__main__":
    sys.path.insert(0, "../../")    
    df = pd.read_csv("../../data/raw/cardio_train.csv", sep=";")
    df_cleaned = clean_data(df)
    x_train, x_test, y_train, y_test, scaler = split_and_scale(df_cleaned)
    
    print("Training all models...\n")
    models = {
        "Logistic Regression": train_logistic_regression(x_train, y_train),
        "Naive Bayes": train_naive_bayes(x_train, y_train),
        "Decision Tree": train_decision_tree(x_train, y_train, max_depth=7),
        "Linear SVM": train_linear_svm(x_train, y_train),
    }
    
    print("Model Performance:\n")
    for model_name, model in models.items():
        metrics = evaluate_model(model, x_test, y_test)
        print(f"{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        # save model
        model_path = f"../../models/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"  Saved to: {model_path}\n")
