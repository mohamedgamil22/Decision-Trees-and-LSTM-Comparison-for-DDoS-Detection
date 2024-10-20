# Comparative Analysis of Traditional and Deep Learning Models for Detecting DDoS Attacks

## Overview

This project aims to compare the performance of traditional machine learning models and deep learning models in detecting Distributed Denial of Service (DDoS) attacks. Using the CIC-DDoS2019 dataset, which contains real-world DDoS traffic, this research evaluates two primary models: Decision Trees (DT) and Long Short-Term Memory (LSTM) networks. The main objective is to analyze the effectiveness of these models in terms of accuracy, F1 score, and false positive rate (FPR), with a particular focus on minimizing false positives to avoid blocking benign traffic.

## CRISP-DM Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, a structured approach for conducting data mining projects. Each notebook in this project is mapped to a specific stage of the CRISP-DM process:

### 1. **Business Understanding**
   - **Goal**: The key objective is to develop and evaluate machine learning models that can detect DDoS attacks with a particular emphasis on reducing false positives. Blocking benign traffic can be costly for businesses, so achieving a balance between accuracy and false positives is critical.
   - **Key Outputs**: A business problem definition and criteria for model evaluation (accuracy, F1 score, and FPR).

### 2. **Data Understanding**
   - **Notebook**: `01 - Data Exploration.ipynb`
   - **Purpose**: This notebook explores the structure, distribution, and characteristics of the CIC-DDoS2019 dataset. 
   - **Main Steps**:
     - Loading and inspecting the dataset to understand its features, dimensions, and types.
     - Visualizing the distribution of attack and benign traffic to identify the class imbalance.
     - Analyzing missing values, infinite values, and other data inconsistencies.
     - Preliminary statistical analysis to understand feature ranges, correlations, and outliers.
     - Plotting histograms, box plots, and other visualizations to examine key features.
   - **Key Outputs**: Insights into the dataset, including identification of null and infinite values, feature distributions, and potential data issues.

### 3. **Data Preparation**
   - **Notebook**: `02 - Data Preprocessing.ipynb`
   - **Purpose**: This notebook performs the necessary data preprocessing to clean the dataset and prepare it for modeling.
   - **Main Steps**:
     - Handling missing values (e.g., replacing null and infinite values based on feature characteristics).
     - Removing duplicate records to reduce redundancy and memory consumption.
     - Encoding categorical variables (e.g., OneHot encoding and binary encoding of ports and protocols).
     - Feature engineering to generate new useful features (e.g., marking well-known and dynamic port ranges).
     - Removing redundant or irrelevant features (e.g., constant columns, sample IDs, unnecessary flags).
     - Converting feature data types for optimization (e.g., category conversion for IPs and protocol).
     - Normalizing and scaling features, particularly important for sequential models like LSTM.
   - **Key Outputs**: A clean, preprocessed dataset ready for modeling, with properly encoded features and relevant statistical characteristics.

### 4. **Modeling**

   - **Notebook**: `03 - DT and RF.ipynb`
   - **Purpose**: This notebook is dedicated to building and evaluating traditional machine learning models, specifically Decision Trees (DT) and Random Forest (RF).
   - **Main Steps**:
     - Splitting the dataset into training and testing sets, ensuring no data leakage between different attack days.
     - Training a Decision Tree model using various hyperparameters (e.g., max_depth, min_samples_split, etc.).
     - Performing hyperparameter tuning using grid search to identify the best model configuration.
     - Evaluating the Decision Tree model on test data and calculating performance metrics such as accuracy, F1 score, and FPR.
     - Training a Random Forest model as a baseline for comparison with Decision Trees.
     - Visualizing feature importance to understand the most influential features in classification.
     - Analyzing the confusion matrix and ROC curve to evaluate model performance.
   - **Key Outputs**: Trained Decision Tree and Random Forest models with corresponding performance metrics (accuracy, F1 score, and FPR).

   - **Notebook**: `04 - LSTM.ipynb`
   - **Purpose**: This notebook implements Long Short-Term Memory (LSTM) networks to handle the sequential nature of the network traffic data.
   - **Main Steps**:
     - Reshaping the data into a sequential format required for LSTM models, maintaining the order of network packets as part of the attack flow.
     - Building the LSTM architecture with configurable layers (0, 1, 2, or 3 hidden layers) and hyperparameters like learning rate and batch size.
     - Training the LSTM model with various configurations and epochs, using techniques such as early stopping and learning rate schedulers.
     - Performing hyperparameter tuning to minimize false positives while maintaining high accuracy.
     - Evaluating the trained LSTM model using accuracy, F1 score, and false positive rate (FPR) to compare with the Decision Tree results.
     - Plotting model performance over epochs, analyzing the learning curves for loss, accuracy, and FPR.
     - Using techniques like oversampling to mitigate class imbalance during training.
   - **Key Outputs**: Trained LSTM models with performance metrics, focusing on the balance between accuracy and false positives in detecting DDoS attacks.

### 5. **Evaluation**
   - **Notebooks**: `03 - DT and RF.ipynb` and  `04 - LSTM.ipynb`
   - **Purpose**: The evaluation phase spans across both the traditional ML and LSTM notebooks. The performance of the models is analyzed in terms of accuracy, F1 score, and false positive rate (FPR), with a detailed comparison to determine the best model.
   - **Main Steps**:
     - Comparing Decision Tree and LSTM models based on the performance metrics.
     - Highlighting the trade-offs between high accuracy and lower false positives, particularly in real-time scenarios where misclassifications could disrupt legitimate traffic.
     - Visualizing and analyzing confusion matrices and ROC curves for a comprehensive comparison of both models.
   - **Key Outputs**: Comparative analysis and a summary of the best-performing models.

### 6. **Deployment** (Future Work)
   - This step is part of the future work discussed in the thesis. The research suggests future improvements by deploying these models in a real-time environment and integrating them into a production intrusion detection system (IDS). There are also opportunities for combining models (e.g., hybrid DT-LSTM models) for optimized performance.

## Notebooks Included

1. `01 - Data Exploration.ipynb`: Initial exploration and analysis of the CIC-DDoS2019 dataset.
2. `02 - Data Preprocessing.ipynb`: Cleaning, feature engineering, and preparation of the dataset for modeling.
3. `03 - DT and RF.ipynb`: Building and evaluating Decision Tree and Random Forest models, including hyperparameter tuning and feature importance analysis.
4. `04 - LSTM.ipynb`: Implementation and evaluation of LSTM deep learning models for detecting DDoS attacks based on sequential network traffic data.

## Dataset

The dataset used in this research is the **CIC-DDoS2019** dataset, which is publicly available from the Canadian Institute for Cybersecurity. It includes network traffic data for both benign and DDoS attack packets, providing a rich dataset for training machine learning models.

