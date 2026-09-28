# Loan-Default_Probability-of-Default-PD-Model
Application-Stage Probability of Default (PD) model for mortgage loans using Python and machine learning. Covers data cleaning, feature engineering, EDA, model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM), calibration, risk segmentation, and leakage-free preprocessing following credit risk best practices.

Loan Default — Probability of Default (PD) Model
An end-to-end credit risk project that explores a mortgage loan dataset and builds a model to estimate a borrower's Probability of Default (PD) at the time of loan application.
Objective
Estimate the likelihood that a borrower will default, using only information available at the time of application — no data generated during underwriting or loan pricing is used. This keeps the model realistic for an actual lending decision, rather than a model that "cheats" using hindsight information.
Assumptions
The prediction is made before loan approval.
Only application-stage information is used as input.
Post-underwriting and pricing variables are excluded.
The model predicts probability of default, not probability of loan approval.
Dataset
148,670 rows × 34 columns
Target: `Status` (1 = Default, 0 = No Default)
112,031 performing loans vs. 36,639 defaulted loans (~24.6% default rate — moderately imbalanced)
13 numerical / 21 categorical columns, covering borrower demographics (Gender, age), loan terms (loan_amount, term, loan_type, loan_purpose), credit profile (Credit_Score, credit_type, co-applicant_credit_type), collateral (property_value, LTV), and affordability (income, dtir1 — debt-to-income ratio).
Project Structure
The notebook (`Loan_Default_PD_Model.ipynb`) follows a linear, ten-step workflow:
Import Libraries
Load & Inspect Data
Data Cleaning & Missing Value Analysis
Univariate Analysis
Bivariate Analysis (features vs. target)
Correlation Analysis
Outlier Detection
Key Insights Summary
Feature Engineering & Encoding
PD Modeling (train, evaluate, interpret)
Key Data Findings
Missing data hotspots: `Upfront_charges` (26.7%), `Interest_rate_spread` (24.6%), `rate_of_interest` (24.5%), `dtir1` (16.2%), `LTV` and `property_value` (~10.2% each).
Leakage risk identified: `rate_of_interest`, `Interest_rate_spread`, and `Upfront_charges` are missing almost exclusively for defaulted loans and are set after underwriting — a live application model would never have them, so they were dropped.
`credit_type` dropped separately: one category (`EQUI`) is an almost perfect proxy for default (~99.99%), which looks like a dataset artifact rather than a genuine credit signal.
Top correlations with default: `dtir1` (+0.078), `income` (−0.065), `property_value` (−0.049), `LTV` (+0.039), `loan_amount` (−0.037) — defaulters tend to have higher debt-to-income and loan-to-value ratios, and lower income and property value.
Feature Engineering
Cross-imputed `property_value` and `LTV` from each other using the algebraic relationship between loan amount, LTV, and property value (row-wise, no leakage).
Added `high_dti` and `high_ltv_flag` binary risk flags (NaN-safe — a missing input yields a missing flag rather than a silent 0).
Added `loan_to_income` and `term_years` derived ratios.
Binned `Credit_Score` into a `credit_score_band` (Very Poor → Exceptional).
Encoded 13 binary categorical fields with explicit mappings, and one-hot encoded the remaining nominal categoricals (Gender, loan_type, loan_purpose, occupancy_type, total_units, Region, age, credit_score_band).
Scaled features for the linear model with `StandardScaler`.
Modeling
Four classifiers were trained on an 80/20 stratified train-test split (class imbalance handled via `class_weight='balanced'` or `scale_pos_weight`):
Model	AUC	Gini	KS	Accuracy	Precision	Recall	F1	Brier
LightGBM ⭐	0.899	0.799	0.665	0.881	0.771	0.735	0.753	0.102
XGBoost	0.899	0.797	0.661	0.879	0.766	0.734	0.749	0.102
Random Forest	0.886	0.773	0.633	0.876	0.777	0.694	0.733	0.117
Logistic Regression	0.776	0.552	0.427	0.727	0.462	0.666	0.546	0.186
ROC-AUC was the primary selection metric, since a PD model needs to rank borrowers by risk accurately rather than just maximize accuracy — accuracy alone can be misleading on an imbalanced credit dataset. LightGBM was selected as the final model.
Final model performance (LightGBM, test set)
```
              precision    recall  f1-score   support
  No Default       0.91      0.93      0.92     22406
     Default       0.77      0.74      0.75      7328
    accuracy                           0.88     29734
```
The model was also checked for calibration (predicted PD vs. observed default rate) and score separation (PD distribution for defaulters vs. non-defaulters), both of which supported that the predicted probabilities are realistic and usable, not just good at ranking.
Risk bands
Borrowers were segmented into five PD bands to demonstrate practical usability:
Band	Count	Avg. Predicted PD	Actual Default Rate
Very Low	3,199	0.077	0.024
Low	11,457	0.169	0.061
Medium	8,095	0.351	0.144
High	2,070	0.603	0.362
Very High	4,913	0.961	0.944
The monotonic increase in actual default rate across bands confirms the PD scores are meaningful and can support real risk-management decisions.
Applications
The resulting PD scores and risk bands can support:
Loan approval decisions
Risk-based pricing
Portfolio monitoring
Credit limit assignment
Expected loss estimation
Tech Stack
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (KNNImputer, OneHotEncoder, StandardScaler, LogisticRegression, RandomForestClassifier, train_test_split, metrics), `xgboost`, `lightgbm`
How to Run
Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm`
Open `Loan_Default_PD_Model.ipynb` in Jupyter.
Run all cells top-to-bottom — the notebook loads the raw data, cleans it, engineers features, trains all four models, and produces the evaluation plots and risk-band table shown above.
Notes & Limitations
`year` is dropped as it's a single constant value (2019) and carries no signal.
The `Security_Type` field contains a known typo in the raw data (`Indriect` instead of `Indirect`), preserved as-is in the binary mapping to match the source data.
Feature engineering and encoding steps are applied consistently to both train and test sets, fit only on the training split, to avoid data leakage.
