import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('SpotifyFeatures.csv', sep=',')
print(data.shape)

# Filter columns
data_filter = data[['genre', 'liveness', 'loudness']]

# Select Pop and Classical genres only
Pop = data_filter[data['genre'] == 'Pop']
Classical = data_filter[data['genre'] == 'Classical']

# Combine Pop and Classical
Pop_Classical = pd.concat([Pop, Classical])
P__C = Pop_Classical.sample(frac=1).reset_index(drop=True)
print(Pop_Classical.shape)
P_C = P__C.replace({'Pop': 0, 'Classical': 1})
print(P_C)

# Randomize dataset

# Features and labels
X = np.array(P_C[['liveness', 'loudness']])
y = np.array(P_C['genre'])

# Feature scaling (manually) - Normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Train-test split (80-20 split)
split_index = int(0.8 * len(y))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index].reshape(-1, 1)
y_test = y[split_index:].reshape(-1, 1)

plt.scatter(Pop_Classical[Pop_Classical['genre'] == 'Pop']['liveness'], 
            Pop_Classical[Pop_Classical['genre'] == 'Pop']['loudness'], 
            color='blue', label='Pop', alpha=0.5)
plt.scatter(Pop_Classical[Pop_Classical['genre'] == 'Classical']['liveness'], 
            Pop_Classical[Pop_Classical['genre'] == 'Classical']['loudness'], 
            color='red', label='Classical', alpha=0.5)

# Adding labels and legend
plt.xlabel('Liveness')
plt.ylabel('Loudness')
plt.legend()
plt.savefig('Task_1d')
plt.show()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights and bias
def initialize_params(n_features):
    weights = np.zeros((n_features, 1))
    bias = 0
    return weights, bias

# Forward propagation
def forward_propagation(X, weights, bias):
    Z = np.dot(X, weights) + bias
    A = sigmoid(Z)
    return A

# Cost function
def compute_cost(A, y):
    m = len(y)
    epsilon = 1e-15
    cost = (-1/m) * np.sum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
    return cost

# Backward propagation
def backward_propagation(X, A, y):
    m = len(y)
    dw = (1/m) * np.dot(X.T, (A - y))
    db = (1/m) * np.sum(A - y)
    return dw, db

# Update parameters
def update_params(weights, bias, dw, db, learning_rate):
    weights -= learning_rate * dw
    bias -= learning_rate * db
    return weights, bias

# Logistic Regression Model
def logistic_regression_model(X_train, y_train, X_test, y_test, num_iterations=25000, learning_rate=0.0001):
    m_train, n_features = X_train.shape
    weights, bias = initialize_params(n_features)
    costs = []
    epochs = []

    for i in range(num_iterations):
        # Forward propagation
        A_train = forward_propagation(X_train, weights, bias)

        # Compute cost
        cost = compute_cost(A_train, y_train)
        costs.append(cost)
        epochs.append(i)

        # Backward propagation
        dw, db = backward_propagation(X_train, A_train, y_train)

        # Update weights and bias
        weights, bias = update_params(weights, bias, dw, db, learning_rate)

        # Print cost every 1000 iterations for monitoring
        if i % 1000 == 0:
            print(f"Iteration {i}, Cost: {cost:.6f}")

    # Make predictions on training and test sets
    train_predictions = predict(X_train, weights, bias)
    test_predictions = predict(X_test, weights, bias)

    return train_predictions, test_predictions, costs, epochs

# Predict function
def predict(X, weights, bias):
    A = forward_propagation(X, weights, bias)
    predictions = (A > 0.5).astype(int)
    return predictions

# Run the logistic regression model
train_predictions, test_predictions, costs, epochs = logistic_regression_model(X_train, y_train, X_test, y_test, num_iterations=10000, learning_rate=0.01)

# Calculate accuracy
train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

# Plot cost over epochs
plt.plot(epochs, costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost reduction over epochs') 
plt.show()

# Print results
print(f"Train Accuracy: {train_accuracy*100:.3f}%")
print(f"Test Accuracy: {test_accuracy*100:.3f}%")

def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

# Compute and display confusion matrix
conf_matrix = compute_confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy from confusion matrix
def calculate_accuracy(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    Pop_accuracy = TN / (TN + FP)
    Classical_accuracy = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, Pop_accuracy, Classical_accuracy

# Calculate and print accuracy
overall_accuracy, pop_accuracy, classical_accuracy = calculate_accuracy(conf_matrix)
print(f"Total Accuracy: {overall_accuracy*100:.3f}%")
print(f"Pop Accuracy: {pop_accuracy*100:.3f}%")
print(f"Classical Accuracy: {classical_accuracy*100:.3f}%")
