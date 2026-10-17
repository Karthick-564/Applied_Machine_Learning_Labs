# 07 · Unsupervised Clustering & Density Estimation

> **Core Focus:** Distance-based geometric partitioning (K-Means), cluster validation metrics (Silhouette & Davies-Bouldin), and soft probabilistic clustering via Gaussian Mixture Models (GMM) with Expectation-Maximization.

---

## 📐 Clustering Mechanics

### 1. K-Means Within-Cluster Sum of Squares (Inertia)
$$\mathcal{J} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

### 2. Gaussian Mixture Models (EM Algorithm)
Models density as a weighted sum of $K$ Gaussian components:
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
* **E-Step:** Compute posterior cluster responsibilities $\gamma_{ik}$.
* **M-Step:** Update mixture weights $\pi_k$, mean vectors $\boldsymbol{\mu}_k$, and covariance matrices $\boldsymbol{\Sigma}_k$.

---

## 📂 Notebook Implementations
* `01_kmeans_clustering_elbow_analysis.ipynb`: Inertia elbow plot, Silhouette score optimization, and 2D spatial cluster visualization.
* `02_gaussian_mixture_models_em.ipynb`: Probabilistic cluster assignments, covariance type evaluation (spherical, diagonal, full), and AIC/BIC model selection.
