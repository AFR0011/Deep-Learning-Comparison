"""
Wine Quality Classification Using GRU-Based Neural Networks

This script implements a machine learning pipeline to classify the quality of red and white wines using a GRU-based neural network in PyTorch.
It leverages advanced data preprocessing, class balancing techniques, and rigorous evaluation to achieve robust model performance.
Dataset link: https://archive.ics.uci.edu/dataset/186/wine+quality

Key Features and Functionalities:

1. **Data Preprocessing**:
   - Loads wine quality datasets ('winequality-red.csv' and 'winequality-white.csv') in CSV format.
   - Maps wine quality labels to numeric classes for classification tasks.
   - Removes highly correlated features to reduce redundancy.
   - Standardizes features using `StandardScaler` and applies PCA for dimensionality reduction.

2. **Class Balancing with SVMSMOTE**:
   - Applies SVMSMOTE to oversample minority classes, addressing class imbalance issues and enhancing classifier performance.

3. **PyTorch GRU Model**:
   - Defines a GRU-based neural network (`WineGRU`) for classification tasks.
   - Incorporates dropout layers for regularization and dynamic learning rate adjustment using the `ReduceLROnPlateau` scheduler.

4. **Cross-Validation and Evaluation**:
   - Implements stratified k-fold cross-validation to assess model performance across multiple data splits.
   - Aggregates confusion matrices across folds to evaluate classification consistency.

5. **Visualization and Metrics**:
   - Visualizes PCA-transformed data for exploratory analysis.
   - Plots normalized confusion matrices for detailed performance insights.
   - Computes evaluation metrics, including accuracy and classification reports, to summarize model efficacy.

6. **Custom Utilities**:
   - Functions for data preprocessing, class weight calculation, synthetic sampling, PCA visualization, and confusion matrix plotting.
   - Encapsulation of model training and evaluation workflows for streamlined cross-validation and testing.

7. **Requirements**:
   - Ensure datasets are saved as 'winequality-red.csv' and 'winequality-white.csv' in the same directory.
   - Assumes wine quality labels range from 3 to 9 and supports multi-class classification with 7 distinct classes.

This script demonstrates the integration of traditional machine learning preprocessing with deep learning techniques to tackle real-world multi-class classification challenges.

Code written by: Ali Farrokhnejad
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE

# Global variables
RED_WINE_PATH = 'winequality-red.csv'
WHITE_WINE_PATH = 'winequality-white.csv'
DELIMITER = ';'
PCA_COMPONENTS = 0.99
RANDOM_STATE = 42
SMOTE_NEIGHBORS = 3
KFOLDS = 6
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-5
DYNAMIC_LR_FACTOR = 0.75
DYNAMIC_LR_PATIENCE = 100

# Load datasets
red_wine = pd.read_csv(RED_WINE_PATH, delimiter=DELIMITER)
white_wine = pd.read_csv(WHITE_WINE_PATH, delimiter=DELIMITER)

# Preprocess data
def preprocessData(data, n_components=None):
    # Extract features and labels
    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    # Map labels from 0-6
    label_map = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
    y_mapped = np.vectorize(label_map.get)(y)

    # Drop highly correlated features
    corr_matrix = data.corr()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_triangle.columns if any(abs(upper_triangle[column]) > 0.9)]
    data = data.drop(columns=high_corr_features)

    # Standardize data using StandardScalar
    X = data.drop(columns=['quality']).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Apply and display PCA results
    if n_components:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

    return X, y_mapped

# Calculate class weights for loss function
def classWeights(y, num_classes=7):
    class_counts = np.bincount(y.numpy(), minlength=num_classes)
    total_samples = len(y)
    class_weights = [total_samples / (num_classes * count) if count else 1.0 for count in class_counts]
    return torch.tensor(class_weights, dtype=torch.float32)

# Apply ADASYN sampling and return balanced data
def synSampling(X, y):
    smoteSampler = SVMSMOTE( random_state=RANDOM_STATE, k_neighbors=SMOTE_NEIGHBORS) 
    X_resampled, y_resampled = smoteSampler.fit_resample(X, y.numpy())
    return torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.long)

# Visualize PCA for 2 most principal components 
def plot_pca_2d(X, y, title="PCA Visualization"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Classes")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Preprocess datasets
X_red, y_red = preprocessData(red_wine, n_components=PCA_COMPONENTS)
X_white, y_white = preprocessData(white_wine, n_components=PCA_COMPONENTS)

# Convert data to PyTorch tensors
X_red, y_red, X_white, y_white = torch.tensor(X_red, dtype=torch.float32), torch.tensor(y_red, dtype=torch.long), torch.tensor(X_white, dtype=torch.float32), torch.tensor(y_white, dtype=torch.long)

# Apply SMOTE sampling to dataset
X_red_balanced, y_red_balanced = synSampling(X_red, y_red)
X_white_balanced, y_white_balanced = synSampling(X_white, y_white)

# Calculate class weights
red_class_weights = classWeights(y_red_balanced)
white_class_weights = classWeights(y_white_balanced)

# Visualize PCA
plot_pca_2d(X_red_balanced, y_red_balanced, title="PCA Visualization for Red Wine Dataset")
plot_pca_2d(X_white_balanced, y_white_balanced, title="PCA Visualization for White Wine Dataset")

# Define GRU model
class WineGRU(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_layers=3, output_dim=7):
        super(WineGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input to add a sequence dimension if necessary
        if x.dim() == 2:  # If input has shape (batch_size, input_dim)
            x = x.unsqueeze(1)  # Reshape to (batch_size, 1, input_dim)

        out, _ = self.gru(x)  # GRU returns output and hidden states

        # out should now have 3 dimensions: (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]  # Use the last GRU output for classification
        out = self.relu(self.fc(out))
        return out

# Training function with learning rate scheduler
def trainModel(model, dataloader, criterion, optimizer, scheduler, epochs=EPOCHS):
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
def evaluateModel(model, dataloader):
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
def plotConfusionMatrix(cm, dataset_name, normalize=False, class_labels=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    if class_labels is None:
        class_labels = np.arange(cm.shape[0])
    
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='YlGnBu', 
                cbar=True, annot_kws={"size": 10}, linewidths=0.5, linecolor='gray',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {dataset_name}', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()

# Cross-validation with confusion matrices
def crossValidate(X, y, class_weights, dataset_name, k=KFOLDS):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
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
        model = WineGRU()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=True)

        # Train model
        trainModel(model, train_loader, criterion, optimizer, scheduler)

        # Evaluate model
        accuracy, cm, report = evaluateModel(model, test_loader)
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
    plotConfusionMatrix(aggregated_cm, dataset_name, normalize=True, class_labels=[3,4,5,6,7,8,9])

# Run cross-validation for red wine dataset
print("Cross-Validation on Red Wine Dataset")
crossValidate(X_red_balanced, y_red_balanced, red_class_weights, "Red Wine")

# Run cross-validation for white wine dataset
print("Cross-Validation on White Wine Dataset")
crossValidate(X_white_balanced, y_white_balanced, white_class_weights, "White Wine")