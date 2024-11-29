"""
Wine Quality Classification Using a Fully Connected Neural Network (FCNN)

This script builds and evaluates a deep learning model to classify the quality of red and white wines using PyTorch. 
It employs rigorous data preprocessing, class balancing, and advanced evaluation techniques to ensure robust model performance.
Dataset link: https://archive.ics.uci.edu/dataset/186/wine+quality

Key Features and Functionalities:

1. **Data Loading and Preprocessing**:
   - Reads red and white wine datasets in CSV format.
   - Maps wine quality labels to numeric classes for classification.
   - Removes highly correlated features to avoid redundancy.
   - Standardizes features and applies Principal Component Analysis (PCA) for dimensionality reduction.

2. **Class Balancing with SVMSMOTE**:
   - Uses Synthetic Minority Oversampling Technique (SVMSMOTE) to address imbalanced classes and improve classifier performance.

3. **PyTorch-based Implementation**:
   - Converts preprocessed datasets into PyTorch tensors for compatibility with deep learning models.
   - Defines a Fully Connected Neural Network (FCNN) with dropout layers for regularization.
   - Implements dynamic learning rate adjustment using ReduceLROnPlateau scheduler.

4. **Visualization**:
   - Visualizes data distribution post-PCA in a 2D space.
   - Plots normalized confusion matrices for detailed performance analysis.

5. **Training and Cross-Validation**:
   - Employs k-fold cross-validation to ensure robust evaluation across data splits.
   - Aggregates confusion matrices across folds to highlight model consistency.

6. **Evaluation Metrics**:
   - Measures accuracy, generates classification reports, and visualizes confusion matrices for comprehensive performance analysis.

7. **Requirements**:
   - Datasets should be saved as 'winequality-red.csv' and 'winequality-white.csv' in a folder named "Data".
   - Designed for datasets with quality labels ranging from 3 to 8 (red wine) and 3 to 9 (white wine).

This script demonstrates the application of machine learning and deep learning techniques for multi-class classification in a practical setting.
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

# Apply SVMSMOTE sampling and return balanced data
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

# Define model (FCNN)
class WineClassifier(nn.Module):
    def __init__(self, input_dim):
        super(WineClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 40)
        self.fc3 = nn.Linear(40, 16)
        self.fc4 = nn.Linear(16, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

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
    aggregated_cm = np.zeros((7, 7), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")
        # Index the PyTorch tensors directly instead of the NumPy arrays
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model, loss, and optimizer
        model = WineClassifier(input_dim=X.shape[1])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=DYNAMIC_LR_FACTOR, patience=DYNAMIC_LR_PATIENCE, verbose=True)

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

