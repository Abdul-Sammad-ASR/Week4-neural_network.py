# AI Week 4 — Neural Networks Basics

A basic **Neural Network implementation** using Python and the Iris flower dataset.

## 📌 Task

Implement a simple feedforward neural network, train it on a dataset, make predictions, and evaluate the results.

## 🧠 Model

The project uses **MLPClassifier (Multi-Layer Perceptron)** from Scikit-learn.

**Architecture:**

```text
Input (4 features)
        ↓
Hidden Layer (8 neurons)
        ↓
Hidden Layer (6 neurons)
        ↓
Output (3 classes)
```

* Activation Function: ReLU
* Optimizer: Adam
* Dataset: Iris Dataset

## 🛠️ Technologies

* Python 3.14
* Scikit-learn
* NumPy

## 🔄 Workflow

1. Load the Iris dataset
2. Scale the input features
3. Split data into training and testing sets
4. Build the neural network
5. Train the model
6. Make predictions
7. Evaluate model performance
8. Predict a new sample

## 📊 Results

* Training iterations: **829**
* Final training loss: **0.0732**
* Test Accuracy: **100.00%**
* Test samples: **30**
* Correct predictions: **30/30**

### New Sample Prediction

Input:

```text
[5.1, 3.5, 1.4, 0.2]
```

Predict
