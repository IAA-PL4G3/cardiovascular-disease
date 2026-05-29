# Analyze num_leaves in LightGBM
import os
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def analyze_lightgbm_leaves(X, y, num_leaves_range=range(10, 101, 10)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_accuracies = []
    test_accuracies = []

    for leaves in num_leaves_range:
        clf = LGBMClassifier(num_leaves=leaves, random_state=42, verbose=-1)
        clf.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    best_train_acc = max(train_accuracies)
    best_test_acc = max(test_accuracies)
    best_train_leaves = list(num_leaves_range)[train_accuracies.index(best_train_acc)]
    best_test_leaves = list(num_leaves_range)[test_accuracies.index(best_test_acc)]
    print(f"Best Train Accuracy: {best_train_acc:.4f} at num_leaves {best_train_leaves}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} at num_leaves {best_test_leaves}")

    plt.figure(figsize=(10, 6))
    plt.plot(num_leaves_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(num_leaves_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Leaves')
    plt.ylabel('Accuracy')
    plt.title('Effect of Number of Leaves on LightGBM Performance')
    plt.xticks(num_leaves_range)
    plt.legend()
    plt.grid()
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/lightgbm_leaves.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # RESULT:
    # Best Train Accuracy: 0.7788 at num_leaves 100
    # Best Test Accuracy: 0.7397 at num_leaves 20
    df = pd.read_csv("../../../data/processed/cardio_train_cleaned.csv", sep=",")
    y = df['cardio']
    X = df.drop(columns=['cardio'])
    analyze_lightgbm_leaves(X, y)


if __name__ == "__main__":
    main()
