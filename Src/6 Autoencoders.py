"""
Wine Quality Classification Pipeline

This script implements a comprehensive workflow for classifying wine quality based on chemical attributes.
It includes data preprocessing, feature engineering, model training, evaluation, and visualization.

Main Components:
1. Data Loading:
   - Loads red and white wine datasets from CSV files.

2. Data Preprocessing:
   - Standardizes feature values.
   - Balances class distributions using SMOTE.

3. Modeling:
   - Defines an Autoencoder for feature extraction and dimensionality reduction.
   - Defines a Classifier for wine quality prediction.

4. Training:
   - Trains the Autoencoder and Classifier sequentially.
   - Uses PyTorch's DataLoader for efficient batch processing.

5. Evaluation:
   - Evaluates models using cross-validation.
   - Computes accuracy, confusion matrices, and classification reports.

6. Visualization:
   - Plots confusion matrices for model evaluation.

7. Class Weight Calculation:
   - Dynamically computes class weights to handle class imbalances during training.

8. Cross-Validation:
   - Performs K-fold cross-validation and aggregates confusion matrices across folds.

Outputs:
- Classification accuracy and performance metrics for both red and white wine datasets.
- Confusion matrices to aid in model interpretability.

Dependencies:
- pandas, numpy, sklearn, PyTorch, matplotlib, seaborn, imbalanced-learn
- Ensure that the required CSV files ('winequality-red.csv', 'winequality-white.csv') are in the working directory.

Notes:
- Customizable SMOTE parameters for handling imbalanced datasets.
- Models and training parameters are configurable for experimentation and optimization.

Usage:
Run the script to preprocess datasets, train models, and evaluate classification performance. 
Customize parameters and experiment with different preprocessing techniques or model architectures as needed.
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
# Apply SMOTE to white wine dataset
X_white_balanced, y_white_balanced = apply_smote(X_white, y_white)

# Calculate weights for red wine
red_class_weights = compute_class_weights(y_red_balanced)
white_class_weights = compute_class_weights(y_white_balanced)

# Define Autoencoder
class WineAutoencoder(nn.Module):
    def __init__(self, input_dim=11, encoding_dim=5):
        super(WineAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define Classifier
class WineClassifier(nn.Module):
    def __init__(self, encoding_dim=5, output_dim=7):  # Adjusted output_dim to 7 classes
        super(WineClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Training function for Autoencoder
def train_autoencoder(autoencoder, dataloader, criterion, optimizer, epochs=200):
    autoencoder.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            _, decoded = autoencoder(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Autoencoder Loss: {running_loss/len(dataloader)}")

# Training function for Classifier
def train_classifier(classifier, autoencoder, dataloader, criterion, optimizer, epochs=200):
    classifier.train()
    autoencoder.eval()  # Freeze autoencoder weights
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            encoded, _ = autoencoder(inputs)
            outputs = classifier(encoded)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Classifier Loss: {running_loss/len(dataloader)}")

# Evaluation function
def evaluate_model(classifier, autoencoder, dataloader):
    classifier.eval()
    autoencoder.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            encoded, _ = autoencoder(inputs)
            outputs = classifier(encoded)
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
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        # Instantiate models
        autoencoder = WineAutoencoder(input_dim=X_train.shape[1], encoding_dim=5)
        classifier = WineClassifier(encoding_dim=5, output_dim=7)

        # Loss functions and optimizers
        autoencoder_criterion = nn.MSELoss()
        classifier_criterion = nn.CrossEntropyLoss(weight=class_weights)
        autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        # Train Autoencoder
        train_autoencoder(autoencoder, train_loader, autoencoder_criterion, autoencoder_optimizer)

        # Train Classifier with encoded features
        train_classifier(classifier, autoencoder, train_loader, classifier_criterion, classifier_optimizer)

        # Evaluate model
        accuracy, cm, report = evaluate_model(classifier, autoencoder, test_loader)
        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}")

        # Pad cm if necessary to match aggregated_cm shape
        cm_shape = cm.shape[0]  # Get the size of cm (assuming square)
        if cm_shape < 7:
            pad_size = (7 - cm_shape) // 2   # Calculate padding size
            cm = np.pad(cm, ((pad_size, pad_size + 1), (pad_size, pad_size + 1)), 'constant')

        # Aggregate confusion matrix
        aggregated_cm += cm
        fold_accuracies.append(accuracy)

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage Accuracy across {k} folds: {avg_accuracy:.4f}")

    # Plot aggregated confusion matrix
    print("\nAggregated Confusion Matrix:")
    plot_confusion_matrix(aggregated_cm, dataset_name)

    return avg_accuracy, aggregated_cm

# Cross-validation for red wine dataset
print("\nRed Wine Dataset - Cross Validation:")
cross_validate(X_red_balanced, y_red_balanced, red_class_weights, "Red Wine")

# Cross-validation for white wine dataset
print("\nWhite Wine Dataset - Cross Validation:")
cross_validate(X_white_balanced, y_white_balanced, white_class_weights, "White Wine")
