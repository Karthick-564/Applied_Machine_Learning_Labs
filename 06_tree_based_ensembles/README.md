# 06 · Ensemble Learning & Class Imbalance Architectures

> **Core Focus:** Advanced ensemble paradigms including Bagging, Boosting (AdaBoost vs. Gradient Boosting), Random Forests, Multi-Tier Stacking, and handling severe class imbalance using SMOTE oversampling.

---

## 🌲 Ensemble Paradigms Compared

```
                ┌─── Bagging (Bootstrap Aggregation) ──► Parallel trees trained on random subsets (Variance Reduction)
                │
                ├─── Boosting (Sequential Learning) ────► Weak learners fit to pseudo-residuals (Bias Reduction)
Ensemble Family ┤
                ├─── Stacking (Meta-Learning) ──────────► Diverse base models blended via a meta-classifier
                │
                └─── Resampling (SMOTE) ────────────────► Synthetic minority vector synthesis for skewed distributions
```

---

## 🧪 Experiments Overview

### 1. `Bagging_Diabetes.ipynb`
* **Dataset:** Pima Indians Diabetes (`diabetes_bagging.csv`).
* **Core Comparison:** Single unconstrained Decision Tree vs. **BaggingClassifier** (10 parallel bootstrap estimators).
* **Finding:** Bagging reduced test variance significantly, boosting classification accuracy by over **$5.2\%$**.

### 2. `Boosting_Churn.ipynb`
* **Dataset:** Telco Customer Churn (`churn_boosting.csv`).
* **Core Comparison:** **AdaBoost** (exponential loss reweighting) vs. **Gradient Boosting** (gradient descent on negative binomial deviance).
* **Diagnostics:** Full **ROC Curve overlay with AUC scores** and Gradient Boosting relative feature importance ranking.

### 3. `Random_Forest_Income.ipynb`
* **Dataset:** Adult Census Income (`income_random_forest.csv`).
* **Architecture:** `RandomForestClassifier` with randomized feature subsampling ($m = \sqrt{p}$).
* **Diagnostics:** Accuracy progression curve plotted as a function of forest size ($B \in [10, 200]$ trees) and Gini impurity feature importance bar plot.

### 4. `Stacking_Heart.ipynb`
* **Dataset:** Clinical Heart Disease (`heart_stacking.csv`).
* **Architecture:** Multi-tier stacking ensemble:
  * *Base Tier:* Logistic Regression, Support Vector Machine, and Decision Tree.
  * *Meta Tier:* Logistic Regression meta-learner trained on out-of-fold probability vectors.
* **Finding:** Stacking successfully combined the linear precision of Logistic Regression with the non-linear flexibility of SVM and Trees.

### 5. `SMOTE_Fraud.ipynb`
* **Dataset:** Credit Card Transaction Fraud (`fraud_smote.csv`).
* **Challenge:** Extreme class imbalance (fraudulent transactions represent $< 1\%$ of observations).
* **Remedy:** **SMOTE (Synthetic Minority Over-sampling Technique)** interpolates synthetic samples along $k$-nearest neighbor feature line segments.
* **Diagnostics:** Before vs. After class count distribution charts, Confusion Matrix, and **Precision-Recall Curve** analysis.

---

## 📊 Benchmark Summary

| Experiment | Dataset | Primary Paradigm | Baseline Score | Ensemble Score | Primary Metric |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **01 · Bagging** | Diabetes | Bagging vs Single Tree | $71.4\%$ | **$77.9\%$** | Accuracy |
| **02 · Boosting** | Customer Churn | AdaBoost vs Gradient Boosting | $0.81$ AUC | **$0.86$ AUC** | ROC-AUC |
| **03 · Random Forest**| Census Income | Forest Size Scaling | $79.2\%$ (10 trees)| **$85.4\%$ (150 trees)**| Accuracy |
| **04 · Stacking** | Heart Disease | Blended Heterogeneous Meta-Model| $81.1\%$ (best single)| **$84.6\%$ (Stacked)** | Accuracy |
| **05 · SMOTE** | Credit Fraud | Minority Synthetic Oversampling | $0.34$ Recall | **$0.88$ Recall** | Minority $F_1$ / PR-AUC |

---

## 📂 Notebook Implementations
* `Bagging_Diabetes.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Bagging_Diabetes.ipynb)
* `Boosting_Churn.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Boosting_Churn.ipynb)
* `Random_Forest_Income.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Random_Forest_Income.ipynb)
* `Stacking_Heart.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Stacking_Heart.ipynb)
* `SMOTE_Fraud.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/SMOTE_Fraud.ipynb)
