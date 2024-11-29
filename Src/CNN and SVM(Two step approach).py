"""
Hybrid CNN-SVM Model for Wine Quality Classification

This script implements a CNN-based feature extractor followed by an SVM classifier to predict wine quality from red and white wine datasets.
It applies advanced preprocessing, dimensionality reduction, class balancing, and cross-validation to ensure robust model evaluation.
Dataset link: https://archive.ics.uci.edu/dataset/186/wine+quality

Key Features:

1. **Data Loading and Preprocessing**:
   - Loads wine datasets from CSV files.
   - Maps wine quality labels to numeric classes for multi-class classification.
   - Removes highly correlated features to reduce redundancy.
   - Standardizes data and supports optional PCA for dimensionality reduction.

2. **Class Balancing**:
   - Balances class distributions using SVMSMOTE to mitigate imbalanced datasets.

3. **Hybrid CNN-SVM Approach**:
   - Uses a CNN to extract meaningful features from wine data.
   - Trains an SVM classifier on these features to predict wine quality.

4. **Cross-Validation**:
   - Employs k-fold cross-validation to ensure reliable performance assessment.
   - Aggregates confusion matrices across folds for comprehensive analysis.

5. **Visualization**:
   - Includes functionality to visualize PCA results in 2D space.
   - Plots confusion matrices with normalization to interpret classification results.

6. **Hyperparameter Tuning**:
   - Implements a learning rate scheduler for dynamic adjustment during CNN training.

7. **Evaluation Metrics**:
   - Computes accuracy, confusion matrix, and classification reports for model evaluation.

Requirements:
   - Red and white wine datasets saved as 'winequality-red.csv' and 'winequality-white.csv', respectively.
   - Libraries: PyTorch, scikit-learn, imbalanced-learn, matplotlib, seaborn.
   - Designed for datasets with quality labels ranging from 3 to 9.

This script showcases a practical application of deep learning and machine learning techniques for wine quality classification.

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
from sklearn.svm import SVC

# Global variables
RED_WINE_PATH = 'winequality-red.csv'
WHITE_WINE_PATH = 'winequality-white.csv'
DELIMITER = ';'
PCA_COMPONENTS = 0.99
RANDOM_STATE = 42
SMOTE_NEIGHBORS = 3
KFOLDS = 6
BATCH_SIZE = 32
EPOCHS = 400
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

    # # Apply and display PCA results
    # if n_components:
    #     pca = PCA(n_components=n_components)
    #     X = pca.fit_transform(X)
    #     print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    #     print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

    return X, y_mapped

# Calculate class weights for loss function
def classWeights(y, num_classes=7):
    class_counts = np.bincount(y.numpy(), minlength=num_classes)
    total_samples = len(y)
    class_weights = [total_samples / (num_classes * count) if count else 1.0 for count in class_counts]
    return torch.tensor(class_weights, dtype=torch.float32)

# Apply ADASYN sampling and return balanced data
def synSampling(X, y):
    smoteSampler = SVMSMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_NEIGHBORS) 
    X_resampled, y_resampled = smoteSampler.fit_resample(X, y.numpy())
    return torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.long)

# # Visualize PCA for 2 most principal components 
# def plot_pca_2d(X, y, title="PCA Visualization"):
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
#     plt.colorbar(scatter, label="Classes")
#     plt.title(title)
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.show()

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

# # Visualize PCA
# plot_pca_2d(X_red_balanced, y_red_balanced, title="PCA Visualization for Red Wine Dataset")
# plot_pca_2d(X_white_balanced, y_white_balanced, title="PCA Visualization for White Wine Dataset")


# CNN Feature Extractor Model
class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        return x


# Training function with learning rate scheduler
def trainFeatureExtractor(model, dataloader, criterion, optimizer, scheduler, epochs=EPOCHS):
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

# Extract features using CNN
def extractFeatures(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            features.append(outputs)
    return torch.cat(features).numpy()  # Convert to numpy for SVM


# SVM training function
def trainSvm(X_train, y_train):
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    return svm

# Evaluate SVM classifier
def evaluateSvm(svm, X_test, y_test, dataset_name):
    predictions = svm.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plotConfusionMatrix(cm, dataset_name, True, [3,4,5,6,7,8,9])
    
    # Classification report
    class_report = classification_report(y_test, predictions, target_names=[str(i) for i in np.unique(y_test)])
    return accuracy, cm, class_report

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

# Cross-validation with CNN-SVM hybrid
def crossValidate(X, y, class_weights, dataset_name, k=KFOLDS):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    aggregated_cm = np.zeros((7, 7), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")
        # Index the PyTorch tensors directly instead of the NumPy arrays
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Reshape X_train and X_test to match 1D convolution input
        X_train = X_train.unsqueeze(1)  # Add channel dimension
        X_test = X_test.unsqueeze(1)  # Add channel dimension

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

        # Initialize feature extractor model
        feature_extractor = FeatureExtractorCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(feature_extractor.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=DYNAMIC_LR_FACTOR, patience=DYNAMIC_LR_PATIENCE, verbose=True)

        # Train CNN as feature extractor
        trainFeatureExtractor(feature_extractor, train_loader, criterion, optimizer, scheduler)

        # Extract features from training and test sets
        X_train_features = extractFeatures(feature_extractor, DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE))
        X_test_features = extractFeatures(feature_extractor, DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE))

        # Train SVM on extracted features
        svm = trainSvm(X_train_features, y_train.numpy())

        # Evaluate SVM on test data
        accuracy, cm, report = evaluateSvm(svm, X_test_features, y_test.numpy(), dataset_name)
        fold_accuracies.append(accuracy)

        # Pad cm (allows for representing red wine cm)
        cm_shape = cm.shape[0] 
        if cm_shape < 7:
            pad_size = (7 - cm_shape) // 2
            cm = np.pad(cm, ((pad_size, pad_size + 1), (pad_size, pad_size + 1)), 'constant')

        # Aggregate cm to total over folds
        aggregated_cm += cm

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
