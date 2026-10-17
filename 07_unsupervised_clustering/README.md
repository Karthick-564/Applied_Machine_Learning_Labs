# 07 · Unsupervised Clustering (K-Means & GMM)

> **Core Focus:** Hard geometric partitioning via K-Means with Elbow and Silhouette validation, compared directly against soft probabilistic density estimation using Gaussian Mixture Models (GMM) with Expectation-Maximization.

---

## 📐 Mathematical Formulation

### 1. K-Means Objective (Within-Cluster Sum of Squares)
$$\mathcal{J} = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
Iteratively alternates between assignment step ($c_i := \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|$) and centroid update step ($\boldsymbol{\mu}_k := \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$).

### 2. Gaussian Mixture Models (Expectation-Maximization)
Models the overall density as a convex combination of $K$ multivariate Gaussian components:
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
* **Expectation (E-Step):** Compute posterior responsibility:
  $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
* **Maximization (M-Step):** Re-estimate mixture weights $\pi_k$, mean vectors $\boldsymbol{\mu}_k$, and covariance matrices $\boldsymbol{\Sigma}_k$.

---

## 🧪 Experiments Overview

### 1. `KMeans_Clustering.ipynb`
* **Dataset:** Customer Mall Spending (`Mall_Customers.csv`).
* **Features:** `Annual Income` vs. `Spending Score`.
* **Diagnostics:**
  * **Elbow Curve:** Evaluates Inertia (WCSS) across $K \in [1, 10]$ to locate the inflection point at $K = 5$.
  * **Silhouette Score:** Quantifies cluster cohesion and separation ($s \approx 0.55$).
  * **2D Cluster Map:** Plots the 5 distinct customer groups with large red 'X' markers denoting final learned centroids $\boldsymbol{\mu}_k$.

### 2. `GMM_Clustering.ipynb`
* **Dataset:** Customer Mall Spending (`Mall_Customers.csv`).
* **Methodology:**
  * **Bayesian Information Criterion (BIC):** Evaluates BIC scores across component counts to avoid overfitting.
  * **Soft Probabilistic Membership:** Plots probability distribution histograms showing confidence profiles across clusters.
  * **GMM Density Contour Plot:** Generates bivariate probability density contours illustrating cluster spread and covariance orientation.
  * **Head-to-Head Comparison:** Visual side-by-side plot comparing hard Voronoi boundaries of K-Means vs. flexible elliptical probability boundaries of GMM.

---

## 📂 Notebook Implementations
* `KMeans_Clustering.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/KMeans_Clustering.ipynb)
* `GMM_Clustering.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/GMM_Clustering.ipynb)
