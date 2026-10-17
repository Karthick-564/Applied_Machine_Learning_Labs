# 05 · K-Nearest Neighbors & Decision Tree Classifiers

> **Core Focus:** Instance-based non-parametric learning via K-Nearest Neighbors (KNN) with decision boundary visualization, and recursive partitioning via Decision Tree Classifiers with structural depth tuning and overfitting diagnostics.

---

## 📐 Algorithmic Mechanics

### 1. K-Nearest Neighbors (KNN)
Given a query sample $\mathbf{x}$, find the $K$ closest training vectors under Euclidean distance:
$$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^p (x_j - x_{i, j})^2}$$
Class label is assigned via majority vote:
$$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = c)$$

### 2. Decision Tree Classifier (CART)
Recursively partitions feature space using Gini impurity to select optimal split points:
$$I_G(t) = 1 - \sum_{k=1}^C p(k \mid t)^2$$
Split quality is measured by impurity reduction:
$$\Delta I_G = I_G(\text{parent}) - \sum_{j} \frac{N_j}{N} I_G(\text{child}_j)$$

---

## 🧪 Experiments Overview

### 1. `KNN_Cancer.ipynb`
* **Dataset:** Wisconsin Diagnostic Breast Cancer ($N = 569$, 30 features).
* **Pipeline:** Feature scaling $\to$ Train/Test split (80/20) $\to$ `KNeighborsClassifier`.
* **Diagnostics:**
  * **Accuracy vs. $K$ Plot:** Grid evaluation over $K \in [1, 25]$ demonstrating the bias-variance tradeoff (low $K$ overfits noise; high $K$ oversmooths boundaries).
  * **2D Decision Boundary:** Meshgrid contour plot visualizing non-linear spatial boundary separating Malignant vs. Benign instances on `Radius` vs. `Texture`.
  * Confusion matrix and classification report ($F_1$-score).

### 2. `Decision_Tree_Loan.ipynb`
* **Dataset:** Bank Loan Approval (`Loan.csv`).
* **Pipeline:** Categorical label encoding $\to$ Handling missing values $\to$ `DecisionTreeClassifier(random_state=42)`.
* **Diagnostics:**
  * **Tree Depth Grid Experiment:** Systematically varying `max_depth` from 1 to 15, comparing Training Score vs. Testing Score to identify the onset of overfitting.
  * **Visual Tree Structure:** Complete graphical tree plot illustrating split criteria, sample distributions, and leaf purity.
  * **Feature Importance Chart:** Horizontal bar chart highlighting the top predictors of loan approval (e.g. `Credit_History`, `ApplicantIncome`).

---

## 📂 Notebook Implementations
* `KNN_Cancer.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_KNN_and_Decision_Trees/KNN_Cancer.ipynb)
* `Decision_Tree_Loan.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_KNN_and_Decision_Trees/Decision_Tree_Loan.ipynb)
