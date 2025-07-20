import numpy as np
import pandas as pd
import math

# Random Seed
np.random.seed(42)

# Load and preprocess the dataset
df = pd.read_csv("/Users/mirhanmetin/CMP4501/CMP_code_asssignment_P3/part2_single_layer_perceptron/Wisconsin_Breast_Cancer_Diagnostic_Dataset.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values.reshape(-1, 1)

# Normalize features
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Manual train-test split
def manual_train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

# Perceptron implementation
class SingleLayerPerceptron:
    def __init__(self, input_dim, hidden_neurons, learning_rate=0.01):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_dim, hidden_neurons) * 0.01
        self.b1 = np.zeros((1, hidden_neurons))
        self.W2 = np.random.randn(hidden_neurons, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output):
        m = X.shape[0]
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=2000):
        for epoch in range(1, epochs+1):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 200 == 0:
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)

# Evaluate average accuracy
def evaluate_average_accuracy(neurons, runs=5):
    accuracies = []
    for _ in range(runs):
        X_train, X_test, y_train, y_test = manual_train_test_split(X, y)
        model = SingleLayerPerceptron(input_dim=X.shape[1], hidden_neurons=neurons, learning_rate=0.01)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    return np.mean(accuracies)

# Run evaluations
print("\n Hidden Neurons Average Accuracy Comparison \n")
for neurons in [2, 4, 8, 16]:
    avg_acc = evaluate_average_accuracy(neurons, runs=5)
    print(f"Hidden Neurons: {neurons} => Average Test Accuracy: {avg_acc * 100:.2f}%")
