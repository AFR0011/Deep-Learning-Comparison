"""
This script implements a Convolutional Neural Network (CNN) for wine quality classification
using red and white wine datasets from the UCI Machine Learning Repository.
Dataset link: https://archive.ics.uci.edu/dataset/186/wine+quality

The script performs data preprocessing, class balancing, model training, and evaluation,
utilizing PyTorch for the deep learning framework and incorporating advanced techniques for
data handling and model optimization.

Key functionalities:

1. Data Loading and Preprocessing:
   - Reads red and white wine datasets from CSV files.
   - Extracts features and maps quality labels to numeric values.
   - Standardizes features and labels.

2. Data Balancing with SMOTE:
   - Addresses class imbalance using the Synthetic Minority Oversampling Technique (SMOTE).
   - Ensures balanced datasets for improved model training.

3. Model Definition and Training:
   - Defines a 1D Convolutional Neural Network (CNN) with two convolutional layers and two fully connected layers for classification.
   - Trains the model with a learning rate scheduler and Adam optimizer for enhanced convergence.

4. Cross-Validation:
   - Implements k-fold cross-validation to evaluate model performance across multiple data splits.
   - Aggregates confusion matrices to provide a comprehensive performance overview.

5. Evaluation and Metrics:
   - Computes accuracy, confusion matrices, and classification reports.
   - Visualizes confusion matrices for interpretability.

6. Visualization:
   - Plots aggregated confusion matrices for cross-validation results.

7. Usage Notes:
   - Requires `winequality-red.csv` and `winequality-white.csv` datasets in the "Data" directory.
   - Handles imbalanced datasets with SMOTE and class-weighted loss functions for fairness.
   - Designed to explore the impact of CNNs on structured tabular datasets.

This script serves as a robust example of applying deep learning techniques to real-world datasets,
focusing on data preparation, balanced training, and reproducibility.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load datasets
red_wine = pd.read_csv('winequality-red.csv', delimiter=";")
white_wine = pd.read_csv('winequality-white.csv', delimiter=";")

# Function to preprocess data
def preprocess_data(data):
    # Extract features and labels
    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    # Map labels 0-6
    label_map = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
    y_mapped = np.vectorize(label_map.get)(y)

    # Standardization
    X = data.drop(columns=['quality']).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_mapped


# Preprocess datasets
X_red, y_red = preprocess_data(red_wine, n_components=0.99)
X_white, y_white = preprocess_data(white_wine, n_components=0.99)

# Convert data to PyTorch tensors
def to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

X_red, y_red = to_tensor(X_red, y_red)
X_white, y_white = to_tensor(X_white, y_white)

# Function to calculate class weights for loss function
def compute_class_weights(y, num_classes=7):
    class_counts = np.bincount(y.numpy(), minlength=num_classes)  # Ensure all classes are counted
    total_samples = len(y)
    class_weights = [total_samples / (num_classes * count) if count else 1.0 for count in class_counts]  # Handle zero counts
    return torch.tensor(class_weights, dtype=torch.float32)

# Function to apply SMOTE and return balanced data
def apply_smote(X, y):
    smote = SMOTE(random_state=42, k_neighbors=1)  # k_neighbors can be tuned
    X_resampled, y_resampled = smote.fit_resample(X, y.numpy())  # Resample X and y
    return torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.long)

# Apply SMOTE to red wine dataset
X_red_balanced, y_red_balanced = apply_smote(X_red, y_red)
# Apply SMOTE to white  wine dataset
X_white_balanced, y_white_balanced = apply_smote(X_white, y_white)

# Calculate weights for red wine
red_class_weights = compute_class_weights(y_red_balanced)
white_class_weights = compute_class_weights(y_white_balanced)

class WineCNN(nn.Module):  # Replaced WineClassifier with WineCNN
    def __init__(self):
        super(WineCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)  # 1 input channel (features), 16 output channels, kernel size 3
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)  # 16 input channels (from conv1), 32 output channels
        self.fc1 = nn.Linear(32 * 7, 64)  # Assuming output of conv2 is 32x7 after flattening - THIS IS THE PROBLEM
        # Calculate the correct output size from the convolutional layers:
        # The output size of a Conv1d layer can be calculated using the formula:
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # Assuming padding=0 and stride=1, for both convolutional layers:
        # conv1_output_size = (X_red_balanced.shape[1] - 3 + 2 * 0) / 1 + 1
        # conv2_output_size = (conv1_output_size - 3 + 2 * 0) / 1 + 1

        # Get the input dimensions
        sample_input = torch.randn(1, 1, X_red_balanced.shape[1]) # Batch size 1, 1 channel, input features

        # Pass through conv layers to get output size
        conv_output_size = self.conv2(self.conv1(sample_input)).shape

        # Flatten the output size for the linear layer
        linear_input_size = conv_output_size[1] * conv_output_size[2]

        # Update the fc1 layer with the correct input size
        self.fc1 = nn.Linear(linear_input_size, 64) # Correctly sized linear layer


        self.fc2 = nn.Linear(64, 7)  # 7 output classes (3-9 mapped to 0-6 for quality)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # Apply first convolution and ReLU
        x = self.relu(self.conv2(x))  # Apply second convolution and ReLU
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))  # Apply FC1 and dropout
        x = self.fc2(x)  # Final output layer
        return x

# Set initial weights of classifier
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Training function with learning rate scheduler
def train_model(model, dataloader, criterion, optimizer, scheduler, epochs=200):
    model.apply(init_weights)
    model.train()
    training_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)
        training_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Plot loss curve
    plt.plot(training_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    # Get unique labels from predictions and true labels
    unique_labels = np.unique(np.concatenate([all_labels, all_predictions]))

    # Filter target names based on unique labels
    target_names = [str(i) for i in unique_labels]

    # Generate classification report with updated target_names
    report = classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0)
    return accuracy, cm, report

# Plot confusion matrix
def plot_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Cross-validation with confusion matrices
def cross_validate(X, y, class_weights, dataset_name, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    aggregated_cm = np.zeros((7, 7), dtype=int)  # Assuming 7 classes

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")
        # Index the PyTorch tensors directly instead of the NumPy arrays
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Reshape X_train and X_test to match 1D convolution input
        X_train = X_train.unsqueeze(1)  # Add channel dimension
        X_test = X_test.unsqueeze(1)  # Add channel dimension

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        # Initialize model, loss, and optimizer
        model = WineCNN()  # Use WineCNN instead of WineClassifier
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=True)

        # Train model
        train_model(model, train_loader, criterion, optimizer, scheduler)

        # Evaluate model
        accuracy, cm, report = evaluate_model(model, test_loader)
        fold_accuracies.append(accuracy)

        # Pad cm if necessary to match aggregated_cm shape
        cm_shape = cm.shape[0]  # Get the size of cm (assuming square)
        if cm_shape < 7:
            pad_size = (7 - cm_shape) // 2   # Calculate padding size
            cm = np.pad(cm, ((pad_size, pad_size + 1), (pad_size, pad_size + 1)), 'constant')

        aggregated_cm += cm  # Now the addition should work

        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")
        print(report)

    # Average accuracy across folds
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\nAverage Accuracy Across Folds: {avg_accuracy * 100:.2f}%")

    # Plot aggregated confusion matrix
    print("\nAggregated Confusion Matrix:")
    plot_confusion_matrix(aggregated_cm, dataset_name)

# Run cross-validation for red wine dataset
print("Cross-Validation on Red Wine Dataset")
cross_validate(X_red_balanced, y_red_balanced, red_class_weights, "Red Wine")

# Run cross-validation for white wine dataset
print("Cross-Validation on White Wine Dataset")
cross_validate(X_white_balanced, y_white_balanced, white_class_weights, "White Wine")
