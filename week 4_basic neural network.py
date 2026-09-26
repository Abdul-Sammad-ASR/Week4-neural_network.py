"""
AI Week 4 Task - Neural Networks Basics
Basic Neural Network Implementation

Dataset : Iris flower dataset (built into scikit-learn)
Model   : Multi-Layer Perceptron (MLP) - a basic feedforward neural network

Workflow:
1. Load dataset
2. Preprocess (scale features - important for neural networks)
3. Split into train/test sets
4. Build & train a neural network (input layer -> hidden layers -> output layer)
5. Make predictions
6. Evaluate & show results
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# -----------------------------
# 1. Load dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("=" * 55)
print("STEP 1: Dataset Loaded")
print("=" * 55)
print(f"Samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
print(f"Classes: {list(target_names)}")

# -----------------------------
# 2. Preprocess: scale features
# Neural networks train better/faster when inputs are on
# a similar scale (mean 0, standard deviation 1).
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Split into train/test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Create & train the neural network
#    Architecture: 4 input features -> 8 neurons -> 6 neurons -> 3 output classes
# -----------------------------
model = MLPClassifier(
    hidden_layer_sizes=(8, 6),   # two hidden layers
    activation="relu",           # non-linear activation function
    max_iter=1000,
    random_state=42
)

print("\n" + "=" * 55)
print("STEP 2: Neural Network Created & Trained")
print("=" * 55)
print("Architecture: Input(4) -> Hidden(8) -> Hidden(6) -> Output(3)")
print("Activation: ReLU | Optimizer: Adam (default)")

model.fit(X_train, y_train)
print(f"Training complete. Iterations run: {model.n_iter_}")
print(f"Final training loss: {model.loss_:.4f}")

# -----------------------------
# 5. Make predictions
# -----------------------------
y_pred = model.predict(X_test)

print("\n" + "=" * 55)
print("STEP 3: Predictions on Test Data")
print("=" * 55)
print(f"{'Actual':<12}{'Predicted':<12}")
print("-" * 24)
for actual, pred in zip(y_test, y_pred):
    print(f"{target_names[actual]:<12}{target_names[pred]:<12}")

# -----------------------------
# 6. Evaluate & show results
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("STEP 4: Results")
print("=" * 55)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# -----------------------------
# 7. Predict a brand-new sample
# -----------------------------
sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("=" * 55)
print("STEP 5: Prediction on a New Sample")
print("=" * 55)
print(f"Input measurements: {sample[0]}")
print(f"Predicted species: {target_names[prediction[0]]}")

# -----------------------------
# 8. Workflow explanation (printed summary)
# -----------------------------
print("\n" + "=" * 55)
print("WORKFLOW EXPLAINED (brief)")
print("=" * 55)
print("""
1. Load data   -> Iris dataset (features = flower measurements, labels = species)
2. Preprocess  -> Scale features so all inputs are on a similar range
3. Split data  -> 80% train, 20% test
4. Build model -> MLP neural network: input layer receives 4 features,
                  two hidden layers (8 and 6 neurons) learn patterns using
                  the ReLU activation function, output layer produces
                  probabilities for the 3 species
5. Train       -> Model adjusts internal weights over many iterations to
                  minimize prediction error (backpropagation + Adam optimizer)
6. Predict     -> Trained model classifies unseen test samples
7. Evaluate    -> Accuracy & classification report show how well it learned
""")
