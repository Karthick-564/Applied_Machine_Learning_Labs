# 03 · Multilinear & Polynomial Regression Models

> **Core Focus:** Ordinary Least Squares (OLS) multilinear regression, coefficient interpretability, residual distribution diagnostics, polynomial curve fitting, and regularized complexity control.

---

## 📐 Mathematical Formulation

### 1. Multiple Linear Regression
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p = \mathbf{w}^T \mathbf{x}$$
Minimizing the residual sum of squares:
$$\mathcal{L}_{\text{OLS}}(\mathbf{w}) = \sum_{i=1}^N \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2$$

### 2. Polynomial Feature Expansion
Mapping scalar feature $x \in \mathbb{R}$ to high-degree vector space:
$$\phi(x) = \left[ 1, \; x, \; x^2, \; x^3, \; \dots, \; x^d \right]^T$$
As polynomial degree $d$ increases, the hypothesis class expands, reducing bias but dramatically increasing variance (overfitting risk).

### 3. Ridge Regularization ($L_2$ Penalty)
$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^p w_j^2$$
Shrinks coefficient magnitudes towards zero, flattening high-degree polynomial oscillations.

---

## 🧪 Experiments Overview

### 1. `Linear_Regression.ipynb`
* **Dataset:** Students Academic Performance (`StudentsPerformance.csv`).
* **Objective:** Predict mathematics assessment scores based on demographic indicators, test preparation courses, parental education levels, and reading/writing scores.
* **Diagnostics:**
  * Extraction and bar-chart visualization of regression coefficient magnitudes ($\mathbf{w}$).
  * Actual vs. Predicted scatter plot with $45^\circ$ ideal identity line.
  * Kernel Density Estimation (KDE) plot of residual errors $e_i = y_i - \hat{y}_i$ to verify normality assumptions.
  * Optimization benchmarks comparing baseline OLS vs. Ridge vs. Lasso.

### 2. `Polynomial_Regression.ipynb`
* **Dataset:** Auto MPG Benchmark (`auto-mpg.data`).
* **Objective:** Model the non-linear relationship between engine horsepower ($x$) and vehicle fuel efficiency (Miles Per Gallon, $y$).
* **Methodology & Diagnostics:**
  * Cleans missing values (handling `?` strings) and performs standard feature scaling.
  * Fits polynomial models of degree $d = 2$, $d = 3$, and $d = 4$.
  * Generates superimposed continuous polynomial curve plots against actual scatter data.
  * Compares Degree 4 unconstrained polynomial against **Ridge Regularized Degree 4** model.
  * Plots **Training MSE vs. Testing MSE across degrees** to isolate the exact point where overfitting occurs.

---

## 📈 Empirical Performance Summary

| Model Configuration | Polynomial Degree ($d$) | Regularization ($\lambda$) | Train MSE | Test MSE | Diagnostic Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Baseline** | 1 | None | High | High | Underfitting; fails to capture curvature |
| **Quadratic** | 2 | None | Moderate | Low | Optimal balance; smooth parabolic fit |
| **Cubic** | 3 | None | Lower | Low | Strong fit; minimal marginal improvement |
| **Quartic (Degree 4)** | 4 | None | **Lowest** | Higher | Overfitting; wild boundary oscillations |
| **Quartic + Ridge** | 4 | $\lambda = 10.0$ | Low | **Lowest** | Regularization suppresses boundary oscillation |

---

## 📂 Notebook Implementations
* `Linear_Regression.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/Linear_Regression.ipynb)
* `Polynomial_Regression.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/Polynomial_Regression.ipynb)
