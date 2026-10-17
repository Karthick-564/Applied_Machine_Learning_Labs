# 10 · Recommendation Systems & Matrix Factorization

> **Core Focus:** User-item interaction matrices, latent semantic factor discovery, Collaborative Filtering via Truncated Singular Value Decomposition (SVD), and Non-Negative Matrix Factorization (NMF).

---

## 📐 Matrix Factorization Algorithms

### 1. Truncated Singular Value Decomposition (SVD)
$$\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
Decomposes the rating matrix $\mathbf{R} \in \mathbb{R}^{m \times n}$ into rank-$k$ orthonormal matrices representing user preference vectors and item property vectors.

### 2. Non-Negative Matrix Factorization (NMF)
$$\min_{\mathbf{W} \ge 0, \mathbf{H} \ge 0} \|\mathbf{R} - \mathbf{W}\mathbf{H}\|_F^2$$
Enforces non-negativity constraints, resulting in strictly additive, highly interpretable parts-based latent representations.

---

## 📂 Notebook Implementations
* `01_collaborative_filtering_truncated_svd.ipynb`: User-item rating matrix generation, top-$k$ latent component projection, and Cosine similarity item recommendations.
* `02_non_negative_matrix_factorization.ipynb`: NMF factorization, reconstruction error calculation (RMSE/MAE), and interpretable latent feature extraction.
