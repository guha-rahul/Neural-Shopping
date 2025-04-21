import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from neural_network import NeuralNetwork

df = pd.read_csv("online_shoppers_intention.csv")

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


