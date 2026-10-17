# 03 · Regression Models & Curve Fitting

> **Core Focus:** Ordinary Least Squares (OLS), Batch Gradient Descent optimization, polynomial degree expansion, regularization, and residual error diagnostics.

---

## 📐 Mathematical Formulation

### 1. Multiple Linear Regression
$$\hat{y} = \mathbf{w}^T \mathbf{x} + b, \quad \mathcal{L}_{\text{MSE}} = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

### 2. Gradient Descent Parameter Update
$$w_j := w_j - \eta \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_{i, j}$$

### 3. Polynomial Expansion
Transforming feature space $\mathbf{x} \mapsto [1, x, x^2, \dots, x^d]$ to capture non-linear relationships while monitoring bias-variance trade-offs.

---

## 📂 Notebook Implementations
* `01_linear_regression_gradient_descent.ipynb`: Manual implementation of Gradient Descent compared directly against Scikit-Learn OLS.
* `02_polynomial_regression_curve_fitting.ipynb`: Polynomial feature expansion, degree selection ($d \in [1, 5]$), and over-fitting analysis via cross-validated $R^2$ scores.
