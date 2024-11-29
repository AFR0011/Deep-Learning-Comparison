# Wine Quality Classification Using Hybrid Models

This repository contains multiple machine learning models for classifying the quality of red and white wines. Currently, the repository includes a hybrid CNN-SVM model, a GRU model, and a Fully Connected Neural Network (FCNN) model. More models will be added in the future to explore various techniques for improving wine quality prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Usage](#usage)
- [License](#license)

## Project Overview
The goal of this project is to classify the quality of red and white wines using different machine learning models. The repository currently features:
1. **Hybrid CNN-SVM Model**: Combines a Convolutional Neural Network (CNN) for feature extraction with a Support Vector Machine (SVM) classifier for wine quality classification.
2. **GRU Model**: Implements a Gated Recurrent Unit (GRU) network for sequence-based modeling of wine quality prediction.
3. **FCNN Model**: A fully connected neural network for direct classification of wine quality from input features.

The project includes preprocessing techniques such as dimensionality reduction (PCA), class balancing with SVMSMOTE, and cross-validation for improved generalization and evaluation.

Future updates will include more models to explore and benchmark different machine learning techniques for this classification task.

## Key Features
- **Data Loading and Preprocessing**:
  - Loads and preprocesses red and white wine datasets (CSV format), including feature standardization, PCA for dimensionality reduction, and class balancing.
  - Maps quality labels to numeric classes for multi-class classification.
  - Removes highly correlated features to reduce redundancy.

- **Class Balancing with SVMSMOTE**:
  - Balances the dataset using Synthetic Minority Oversampling Technique (SVMSMOTE) to address class imbalance.

- **Hybrid CNN-SVM Model**:
  - A CNN model extracts features from the data, which are then used to train an SVM classifier for wine quality classification.
  
- **GRU Model**:
  - Implements a Gated Recurrent Unit (GRU) model, designed for handling sequential data, for wine quality prediction.

- **FCNN Model**:
  - A Fully Connected Neural Network (FCNN) is used to predict wine quality directly from the features, providing another perspective on model performance.

- **Cross-Validation**:
  - k-fold cross-validation is used to evaluate the models on multiple data splits for consistent performance assessment.

- **Visualization**:
  - PCA results are visualized to understand feature space distribution.
  - Confusion matrices are visualized for performance analysis of the classifiers.

- **Hyperparameter Tuning**:
  - Dynamic learning rate adjustment with a scheduler during training for the CNN and FCNN models.

## Requirements
To run this project, you will need the following Python libraries:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- torch (PyTorch)
- matplotlib
- seaborn

You can install the required libraries using `pip`:
```bash
pip install -r requirements.txt
```
## File Structure
The repository contains the following files:

/
├── README.md                   # Project overview and documentation
├── Data                        # Contains red and white wine datasets in .csv format
├── Src                         # Contains implementation of different architectures and models
├── requirements.txt            # List of required Python packages
└── LICENSE                     # Project license

## Usage
**Clone the Repository:** To clone this repository, use the following command:


```
git clone https://github.com/AFR0011/Deep-Learning-Comparison.git
cd Deep-Learning-Comparison
```

**Prepare the Data:** Make sure you have the datasets *winequality-red.csv* and *winequality-white.csv* available in the project directory.

**Run the Models:** You can find the models in the *Src* directory.

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
