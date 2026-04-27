# Analyze estimators on random forest
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def analyze_random_forest_estimators(X, y, n_estimators_range=range(10, 201, 10)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_accuracies = []
    test_accuracies = []

    for n in n_estimators_range:
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
        clf.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # print best accuracies and corresponding n_estimators
    best_train_acc = max(train_accuracies)
    best_test_acc = max(test_accuracies)
    best_train_n = n_estimators_range[train_accuracies.index(best_train_acc)]
    best_test_n = n_estimators_range[test_accuracies.index(best_test_acc)]
    print(f"Best Train Accuracy: {best_train_acc:.4f} at n_estimators {best_train_n}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} at n_estimators {best_test_n}")
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(n_estimators_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Estimators in Random Forest')
    plt.ylabel('Accuracy')
    plt.title('Effect of Number of Estimators on Random Forest Performance')
    plt.xticks(n_estimators_range)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # RESULT:
    # Best Train Accuracy: 1.0000 at n_estimators 100
    # Best Test Accuracy: 0.7278 at n_estimators 80
    df = pd.read_csv("../../data/processed/cardio_train_cleaned.csv", sep=",")
    y = df['cardio']
    X = df.drop(columns=['cardio'])
    analyze_random_forest_estimators(X, y)

if __name__ == "__main__":
    main()
