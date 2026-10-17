# 05 · Support Vector Machines (SVM) & Kernel Methods

> **Core Focus:** Maximum margin hyperplanes, Lagrangian dual formulation, slack variable penalty ($C$), and non-linear mapping using the Radial Basis Function (RBF) kernel.

---

## 📐 Mathematical Framework

### Primal Soft-Margin Optimization
$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

### The Kernel Trick: Radial Basis Function (RBF)
Implicitly maps inputs to an infinite-dimensional Hilbert space without computing coordinates directly:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

---

## 📂 Notebook Implementations
* `01_linear_svm_margin_maximization.ipynb`: Margin boundary visualization, support vector extraction, and sensitivity to penalty parameter $C$.
* `02_kernel_svm_rbf_nonlinear_boundaries.ipynb`: Non-linear decision boundary synthesis, tuning RBF kernel parameter $\gamma$, and multi-class classification.
