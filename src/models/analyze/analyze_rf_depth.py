# Analyze max_depth in random forest
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def analyze_random_forest_depth(X, y, max_depth_range=range(1, 21)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_accuracies = []
    test_accuracies = []

    for depth in max_depth_range:
        clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        clf.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    best_train_acc = max(train_accuracies)
    best_test_acc = max(test_accuracies)
    best_train_depth = list(max_depth_range)[train_accuracies.index(best_train_acc)]
    best_test_depth = list(max_depth_range)[test_accuracies.index(best_test_acc)]
    print(f"Best Train Accuracy: {best_train_acc:.4f} at depth {best_train_depth}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} at depth {best_test_depth}")

    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Max Depth of Random Forest')
    plt.ylabel('Accuracy')
    plt.title('Effect of Max Depth on Random Forest Performance')
    plt.xticks(max_depth_range)
    plt.legend()
    plt.grid()
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/rf_depth.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # RESULT:
    # Best Train Accuracy: 0.9133 at depth 20
    # Best Test Accuracy: 0.7399 at depth 8
    df = pd.read_csv("../../../data/processed/cardio_train_cleaned.csv", sep=",")
    y = df['cardio']
    X = df.drop(columns=['cardio'])
    analyze_random_forest_depth(X, y)


if __name__ == "__main__":
    main()
