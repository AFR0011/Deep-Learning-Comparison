"""
This script processes and classifies wine quality data using machine learning techniques in Python. It integrates data preprocessing, model training, evaluation, and visualization. 

Modules and Functions:
- Data Loading:
  - Reads red and white wine datasets using Pandas.
  - Datasets: 'winequality-red.csv' and 'winequality-white.csv'.
- Data Preprocessing:
  - `preprocess_data(data)`: Extracts features, standardizes them.
  - `to_tensor(X, y)`: Converts NumPy arrays to PyTorch tensors.
  - `apply_smote(X, y)`: Balances classes using SMOTE (Synthetic Minority Oversampling Technique).
  - `compute_class_weights(y, num_classes=7)`: Computes class weights for imbalanced datasets.
- Visualization:
  - `plot_confusion_matrix(cm, dataset_name)`: Plots confusion matrices using Seaborn.
- Model Definition:
  - `WineLSTM`: LSTM-based neural network for wine quality classification.
  - Includes features like dropout for regularization and ReLU activation.
  - `init_weights(m)`: Initializes weights for linear layers using Kaiming initialization.
- Training and Evaluation:
  - `train_model(model, dataloader, criterion, optimizer, scheduler, epochs)`: Trains the LSTM model with a learning rate scheduler and visualizes the training loss.
  - `evaluate_model(model, dataloader)`: Evaluates the trained model, calculates accuracy, confusion matrix, and classification report.
- Cross-Validation:
  - `cross_validate(X, y, class_weights, dataset_name, k)`: Implements k-fold cross-validation, tracks accuracy across folds, and aggregates confusion matrices.

Key Features:
- Balances datasets with SMOTE to address class imbalance.
- Implements a sequential model (LSTM) for temporal or contextual patterns.
- Tracks model performance with confusion matrices, classification reports, and accuracy metrics.
- Employs cross-validation for robust evaluation.

Applications:
This script is tailored for wine quality classification but can be adapted for other multi-class classification problems with imbalanced datasets.

Outputs:
- Accuracy metrics and confusion matrices for both datasets (red and white wine).
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
X_red, y_red = preprocess_data(red_wine)
X_white, y_white = preprocess_data(white_wine)

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

# Define LSTM model with dropout
class WineLSTM(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=2, output_dim=7, dropout_prob=0.5): # Change input_dim to 10
        super(WineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input to be [batch_size, sequence_length, input_dim]
        # Assuming sequence_length is 1 in this case
        x = x.view(x.size(0), 1, x.size(1))
        out, _ = self.lstm(x)  # LSTM returns output and hidden states
        out = out[:, -1, :]  # Take the last output of the sequence for classification
        out = self.dropout(self.relu(self.fc(out)))
        return out

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

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        # Initialize model, loss, and optimizer
        model = WineLSTM()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100, verbose=True)

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