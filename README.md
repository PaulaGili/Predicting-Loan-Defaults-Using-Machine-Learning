# Predicting Loan Defaults Using Machine Learning

This project develops a machine learning model to predict whether a personal loan will be repaid or defaulted.

## Objective

To proactively assess credit risk by building a binary classification model (`paid = 1` / `default = 0`). The goal is to aid financial institutions in making informed lending decisions and minimizing exposure to loan defaults.

---

## Methodology

The project follows a comprehensive machine learning pipeline including the following steps:

1. **Exploratory Data Analysis (EDA)**
   - Summary statistics and data overview
   - Visualizations of variable distributions and correlations
   - Identification of outliers and behavioral patterns
   - Generated reports: `output_graphs.pdf` and `distribution_variables.pdf`

2. **Data Preprocessing**
   - Handling missing or inconsistent values
   - Outlier removal
   - Encoding categorical features
   - Scaling numerical variables to enhance model performance

3. **Addressing Class Imbalance**
   - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset by oversampling the minority class (`default`)

4. **Model Development**
   - Training and tuning various supervised classifiers:
     - Random Forest  
     - XGBoost  
     - HistGradientBoosting  
     - Logistic Regression

5. **Model Evaluation**
   - Metrics employed:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - Confusion matrix for detailed error analysis

6. **Model Interpretability**
   - Feature importance ranking
   - Local and global explanations using **SHAP** values

---

## Repository Structure

    - Data.txt/ # Input data
    - Distribution in variables.pdf # EDA: distribution and analysis of variables
    - output_graphs.pdf # Plots and graphs from exploratory analysis
    - FINANCIAL_DEFAULT_PREDICTION.ipynb/  # Jupyter Notebook structured by each stage of the ML pipeline
    - Results.pdf/ # Model outputs, metrics, and reports
    - README.md # This fil
    - requirements.txt # List of Python dependencies

---
