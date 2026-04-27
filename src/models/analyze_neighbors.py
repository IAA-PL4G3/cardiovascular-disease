# Analyze neighbors in KNN
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def analyze_knn_neighbors(X, y, neighbors_range=range(1, 21)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_accuracies = []
    test_accuracies = []

    for n_neighbors in neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, knn.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, knn.predict(X_test_scaled))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # print best accuracies and corresponding neighbors
    best_train_acc = max(train_accuracies)
    best_test_acc = max(test_accuracies)
    best_train_neighbors = neighbors_range[train_accuracies.index(best_train_acc)]
    best_test_neighbors = neighbors_range[test_accuracies.index(best_test_acc)]
    print(f"Best Train Accuracy: {best_train_acc:.4f} at n_neighbors {best_train_neighbors}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} at n_neighbors {best_test_neighbors}")

    plt.figure(figsize=(10, 6))
    plt.plot(neighbors_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(neighbors_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.title('Effect of Number of Neighbors on KNN Performance')
    plt.xticks(neighbors_range)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # RESULT:
    # Best Train Accuracy: 1.0000 at n_neighbors 1
    # Best Test Accuracy: 0.7245 at n_neighbors 19
    df = pd.read_csv("../../data/processed/cardio_train_cleaned.csv", sep=",")
    y = df['cardio']
    X = df.drop(columns=['cardio'])
    analyze_knn_neighbors(X, y)

if __name__ == "__main__":
    main()
