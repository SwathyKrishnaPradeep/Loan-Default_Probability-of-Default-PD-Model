# Loan-Default_Probability-of-Default-PD-Model
Application-Stage Probability of Default (PD) model for mortgage loans using Python and machine learning. Covers data cleaning, feature engineering, EDA, model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM), calibration, risk segmentation, and leakage-free preprocessing following credit risk best practices.

# Loan Default Prediction (Probability of Default - PD Model)

## Project Overview

This project develops an **Application Probability of Default (PD) Model** to estimate the likelihood of a borrower defaulting on a loan using only information available at the time of application.

The objective is to build a realistic and interpretable credit risk model while following industry best practices to prevent data leakage and ensure that predictions can be made before a lending decision.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, and interpretation.

---

## Business Objective

Financial institutions rely on Probability of Default (PD) models to evaluate credit risk during loan underwriting.

This project aims to:

* Predict whether an applicant is likely to default
* Identify the most influential risk drivers
* Build an interpretable model suitable for credit risk applications
* Demonstrate an end-to-end PD modelling workflow

---

## Dataset

The dataset contains historical loan application records with applicant demographics, financial characteristics, property information, and loan details.

Target Variable:

* **Status**

  * 0 → Non-default
  * 1 → Default

Only variables available at the application stage were retained.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Data quality assessment
* Missing value analysis
* Distribution analysis
* Outlier detection
* Class balance analysis
* Correlation analysis
* Feature relationship analysis

---

### 2. Data Cleaning

* Removed duplicate records
* Treated missing values
* Corrected inconsistent categories
* Removed leakage-prone variables
* Eliminated variables unavailable during loan application

---

### 3. Feature Engineering

* Created application-stage features
* Grouped categorical variables
* Encoded categorical features
* Prepared model-ready dataset

---

### 4. Train-Test Split

Feature engineering and encoding were fitted **only on the training data** to prevent data leakage.

---

### 5. Model Development

Models implemented include:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

---

### 6. Model Evaluation

Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

Model thresholds were also analysed across multiple probability cut-offs to understand business trade-offs.

---

### 7. Model Interpretation

The project includes:

* Feature importance
* Business interpretation of important variables
* Risk segmentation
* Discussion of model assumptions and limitations

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Repository Structure

```
Loan_Default_PD_Model/

│
├── Loan_Default_PD_Model.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv (optional)
│
└── images/
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## Key Highlights

* End-to-end Probability of Default (PD) modelling project
* Application-stage modelling to prevent target leakage
* Comprehensive exploratory data analysis
* Leakage-aware preprocessing pipeline
* Multiple machine learning models compared
* Model performance evaluated using industry-standard metrics
* Business-focused interpretation of predictions

---

## Future Improvements

* Hyperparameter tuning
* Probability calibration
* Cross-validation
* SHAP explainability
* Population Stability Index (PSI)
* Model monitoring framework
* Scorecard development

---

## Disclaimer

This project was developed for educational and portfolio purposes. It demonstrates the complete workflow of an application-stage credit risk model and does not represent a production-ready banking model.

---

## Author

**Swathy Krishna Pradeep**

M.Sc. Economics | Credit Risk Analytics | Risk Modelling | Python | SQL | Machine Learning

Open to opportunities in Credit Risk, Model Risk Management, and Risk Analytics.

