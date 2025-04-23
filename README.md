# 🧠 ICU Readmission Prediction Using Machine Learning

## 📄 Project Overview
This thesis investigates the application of machine learning techniques to predict unplanned **Intensive Care Unit (ICU) readmissions**, a critical issue in patient safety and hospital management. The goal is to support **clinical decision-making** and improve healthcare efficiency through accurate predictive modeling.

---

## 🎯 Objective
To build and evaluate classification models that can identify patients at risk of readmission to the ICU shortly after discharge, using structured clinical data from real-world hospital records.

---

## 🗂️ Data Source
- **Dataset**: MIMIC-III (Medical Information Mart for Intensive Care)
- **Content**: De-identified data from over 40,000 ICU patients
- **Preprocessing**:
  - Feature selection guided by clinical relevance
  - Imputation for missing values
  - Normalization and encoding of variables

---

## 🔍 Methodology
A comparative analysis was conducted using the following algorithms:

| Algorithm             | Description                                      |
|-----------------------|--------------------------------------------------|
| Logistic Regression   | Baseline linear model                           |
| Random Forest         | Ensemble method using decision trees            |
| XGBoost               | Gradient boosting-based model                   |
| Support Vector Machine (SVM) | Hyperplane-based classifier             |
| K-Nearest Neighbors (KNN) | Instance-based learning algorithm          |
| Multi-layer Perceptron (MLP) | Neural network-based approach           |

> **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## ✅ Selected Model for Deployment
The **Logistic Regression** coefficients was selected for deployment based on its consistent performance across multiple evaluation metrics and its interpretability in clinical settings.

---

## 🚀 Streamlit Application
An interactive **Streamlit app** was developed to operationalize the model:

- 🧾 User inputs clinical features
- 📊 Predicts ICU readmission risk in real-time
- 🔄 Designed for use by clinicians and researchers

