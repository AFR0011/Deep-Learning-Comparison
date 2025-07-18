import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import torch
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
    smoteSampler = SVMSMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_NEIGHBORS) 
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

# SVM training function
def trainSvm(X_train, y_train):
    svm = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE)
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
    plotConfusionMatrix(cm, dataset_name, True, [3, 4, 5, 6, 7, 8, 9])
    
    # Classification report
    class_report = classification_report(y_test, predictions, target_names=[str(i) for i in np.unique(y_test)])
    return accuracy, cm, class_report

# Plot confusion matrix
def plotConfusionMatrix(cm, dataset_name, normalize=False, class_labels=None):
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = cm.astype('float') / np.where(cm_sum == 0, 1, cm_sum)
    
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
# Cross-validation with SVM
def crossValidate(X, y, class_weights, dataset_name, k=KFOLDS):
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    aggregated_cm = np.zeros((7, 7), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train, X_test = X[train_idx].numpy(), X[test_idx].numpy()
        y_train, y_test = y[train_idx].numpy(), y[test_idx].numpy()

        # Train SVM on training set
        svm = trainSvm(X_train, y_train)

        # Evaluate SVM on test set
        accuracy, cm, report = evaluateSvm(svm, X_test, y_test, dataset_name)

        # Pad cm (allows for representing red wine cm)
        cm_shape = cm.shape[0] 
        if cm_shape < 7:
            pad_size = (7 - cm_shape) // 2
            cm = np.pad(cm, ((pad_size, pad_size + 1), (pad_size, pad_size + 1)), 'constant')

        # Aggregate cm to total over folds
        aggregated_cm += cm

        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")
        print(report)
    
    # Average accuracy across folds
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\nAverage Accuracy Across Folds: {avg_accuracy * 100:.2f}%")

    # Plot aggregated confusion matrix
    print("\nAggregated Confusion Matrix:")
    plotConfusionMatrix(aggregated_cm, dataset_name, normalize=True, class_labels=[3, 4, 5, 6, 7, 8, 9])

# Run cross-validation for red wine dataset
print("Cross-Validation on Red Wine Dataset")
crossValidate(X_red_balanced, y_red_balanced, red_class_weights, "Red Wine")

# Run cross-validation for white wine dataset
print("Cross-Validation on White Wine Dataset")
crossValidate(X_white_balanced, y_white_balanced, white_class_weights, "White Wine")
