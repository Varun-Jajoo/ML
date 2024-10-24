import numpy as np
import pandas as pd

# Step 1: Load the Iris dataset
iris = "Iris.csv"
columns = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
df = pd.read_csv(iris)

# Step 2: Remove the 'Id' column and map species to numerical labels
df = df.drop(columns=['Id'])
df['Species'] = df['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

# Step 3: Prepare the data
X = df.iloc[:, :-1].values.astype(float)  # Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
y = df['Species'].values  # Target: Species (already numeric)

# Step 4: Add bias term (intercept) to the feature matrix
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add a column of ones for bias

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hypothesis function
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Step 5: Cost function and gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        theta -= (alpha / m) * np.dot(X.T, (hypothesis(X, theta) - y))
    return theta

# One-vs-All Logistic Regression
def one_vs_all(X, y, num_labels, alpha, iterations):
    all_theta = np.zeros((num_labels, X.shape[1]))
    for i in range(num_labels):
        y_i = np.where(y == i, 1, 0)  # Create binary labels for class i
        all_theta[i] = gradient_descent(X, y_i, np.zeros(X.shape[1]), alpha, iterations)
    return all_theta

# Step 6: Predict function
def predict_one_vs_all(all_theta, X):
    return np.argmax(sigmoid(np.dot(X, all_theta.T)), axis=1)

# Step 7: Train the model and evaluate
alpha = 0.1
iterations = 1000
num_labels = 3  # Number of unique species in the dataset

all_theta = one_vs_all(X, y, num_labels, alpha, iterations)
predictions = predict_one_vs_all(all_theta, X)
accuracy = np.mean(predictions == y) * 100

print(f"Accuracy: {accuracy:.2f}%")
