import joblib
import pandas as pd
import numpy as np

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
        return np.mean(y_pred.flatten() == y)


def predict_purchase(input_values):
    try:
        model = joblib.load('trained_nn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')

        input_df = pd.DataFrame([input_values])
        input_df = pd.get_dummies(input_df, columns=['Month', 'VisitorType'], drop_first=True)

        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[feature_names]

        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)

        return bool(prediction[0])

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None


if __name__ == "__main__":
    sample_input = {
        'Administrative': 3,
        'Administrative_Duration': 87.33,
        'Informational': 0,
        'Informational_Duration': 0,
        'ProductRelated': 27,
        'ProductRelated_Duration': 798.33,
        'BounceRates': 0,
        'ExitRates': 0.012644,
        'PageValues': 22.91,
        'SpecialDay': 0.8,
        'Month': 'Feb',
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 3,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': 0
    }

    result = predict_purchase(sample_input)
    print(f"Will make purchase: {result}")
