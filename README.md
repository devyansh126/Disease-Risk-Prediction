# Multi-Disease Risk Prediction System

An end-to-end machine learning pipeline built to evaluate and predict the risk profiles of multiple chronic health conditions—specifically **Diabetes, Heart Disease, Obesity, and Sleep Apnea**—using clinical patient data.

This repository focuses on applying advanced statistical analysis, robust data preprocessing, and rigorous classification techniques while strictly preventing common data science pitfalls like target leakage and training data contamination.

## 🚀 Key Features & Engineering Implementations

* **Target Leakage Mitigation:** Ensured clinical integrity by identifying and removing strong proxy variables (such as stripping height and weight metrics from obesity tracking models to ensure the model evaluates behavioral/demographic risks rather than trivially reconstructing BMI).
* **Strict Pipeline Isolation:** Implemented structured data processing where outlier removal (via a conservative $3 \times \text{IQR}$ threshold) and missing value imputations are calculated exclusively from the training splits to guarantee zero data leakage into the validation sets.
* **Class Imbalance Optimization:** Leveraged an adaptive resampling framework using SMOTE to handle minority class distribution challenges selectively based on the precise imbalance ratio of each condition.
* **Precision-Recall Threshold Tuning:** Programmed a fine-grained, 0.05-step probability threshold search optimized for medical classification logic, moving beyond default 0.5 cutoffs to maximize target class recall and aggressively reduce critical false negatives.
* **Unsupervised Patient Clustering:** Integrated an exploratory pipeline combining Principal Component Analysis (PCA) and K-Means silhouette evaluation to identify structural similarities and groupings within non-annotated patient segments.
* **Comprehensive Analytics Dashboard:** Built a visualization framework to export standard performance metrics (Test Accuracy, F1-Score, Cross-Validation Means, and ROC-AUC curves) alongside structural decision boundary profiling using Support Vector Machines (SVM).

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Machine Learning & Frameworks:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Data Pipelines & Structuring:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
