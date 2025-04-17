import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv("D:\Downloads\ml project database\online_shoppers_intention.csv")

print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nData summary:")
print(df.describe())

print("\nTarget variable (Revenue) distribution:")
print(df['Revenue'].value_counts())
print(df['Revenue'].value_counts(normalize=True))

df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
X = df.drop('Revenue', axis=1)
y = df['Revenue']
X_np = X.values
y_np = y.values
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.1)
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * 0.1)
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.layer_outputs = []
        self.layer_inputs = []
        layer_input = X
        self.layer_inputs.append(layer_input)
        layer_output = self.sigmoid(np.dot(layer_input, self.weights[0]) + self.biases[0])
        self.layer_outputs.append(layer_output)
        for i in range(1, len(self.hidden_sizes)):
            layer_input = self.layer_outputs[-1]
            self.layer_inputs.append(layer_input)
            layer_output = self.sigmoid(np.dot(layer_input, self.weights[i]) + self.biases[i])
            self.layer_outputs.append(layer_output)
        layer_input = self.layer_outputs[-1]
        self.layer_inputs.append(layer_input)
        output = self.sigmoid(np.dot(layer_input, self.weights[-1]) + self.biases[-1])
        self.layer_outputs.append(output)
        return self.layer_outputs[-1]
    
    def backward(self, X, y, output):
        y_reshaped = y.reshape(-1, 1)
        output_error = y_reshaped - output
        delta = output_error * self.sigmoid_derivative(output)
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        weight_gradients[-1] = np.dot(self.layer_inputs[-1].T, delta)
        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.hidden_sizes), 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.layer_outputs[i-1])
            weight_gradients[i-1] = np.dot(self.layer_inputs[i-1].T, delta)
            bias_gradients[i-1] = np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * weight_gradients[i]
            self.biases[i] += self.learning_rate * bias_gradients[i]
    
    def train(self, X, y, epochs=1000, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        loss_history = []
        for epoch in range(1, epochs + 1):
            total_loss = 0
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                output = self.forward(X_batch)
                epsilon = 1e-15
                y_batch_reshaped = y_batch.reshape(-1, 1)
                batch_loss = -np.mean(
                    y_batch_reshaped * np.log(output + epsilon) + 
                    (1 - y_batch_reshaped) * np.log(1 - output + epsilon)
                )
                total_loss += batch_loss * len(batch_indices)
                self.backward(X_batch, y_batch, output)
            avg_loss = total_loss / n_samples
            loss_history.append(avg_loss)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        return loss_history
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred.flatten() == y)
        return accuracy

input_size = X_train_scaled.shape[1]
hidden_sizes = [16, 8]
output_size = 1

nn = NeuralNetwork(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    learning_rate=0.01
)

loss_history = nn.train(
    X_train_scaled, 
    y_train, 
    epochs=500, 
    batch_size=16, 
    verbose=True
)

print("\nSaving model and preprocessing components...")
joblib.dump(nn, 'trained_nn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("""
Saved:
- trained_nn_model.pkl (Trained neural network)
- scaler.pkl (Feature scaler)
- feature_names.pkl (Feature name list)
""")

train_accuracy = nn.evaluate(X_train_scaled, y_train)
test_accuracy = nn.evaluate(X_test_scaled, y_test)

print(f"\nTrain accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

y_pred_proba = nn.predict_proba(X_test_scaled)
y_pred = nn.predict(X_test_scaled)

true_positives = np.sum((y_pred.flatten() == 1) & (y_test == 1))
false_positives = np.sum((y_pred.flatten() == 1) & (y_test == 0))
false_negatives = np.sum((y_pred.flatten() == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")


