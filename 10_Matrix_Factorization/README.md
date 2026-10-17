# 10 · Model-Based Collaborative Filtering & Matrix Factorization

> **Core Focus:** Latent factor recommendation models decomposing large sparse user-item interaction matrices using Truncated Singular Value Decomposition (SVD) and Non-Negative Matrix Factorization (NMF) on the MovieLens 100k benchmark.

---

## 📐 Matrix Factorization Mathematics

### 1. Truncated Singular Value Decomposition (SVD)
Approximates the rating matrix $\mathbf{R} \in \mathbb{R}^{m \times n}$ by its rank-$k$ low-rank projection:
$$\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
Where:
* $\mathbf{U}_k \in \mathbb{R}^{m \times k}$ contains orthogonal user latent affinity vectors.
* $\mathbf{\Sigma}_k \in \mathbb{R}^{k \times k}$ contains the top $k$ singular values (factor strength).
* $\mathbf{V}_k \in \mathbb{R}^{n \times k}$ contains orthogonal item latent property vectors.

### 2. Non-Negative Matrix Factorization (NMF)
Constrains both user and item factor matrices to non-negative values ($\ge 0$):
$$\min_{\mathbf{W} \ge 0, \mathbf{H} \ge 0} \|\mathbf{R} - \mathbf{W}\mathbf{H}\|_F^2 = \sum_{u=1}^m \sum_{i=1}^n \left( R_{ui} - \sum_{k=1}^K W_{uk} H_{ki} \right)^2$$
Because subtraction is prohibited, factors are strictly additive, yielding intuitive parts-based interpretation (e.g. genre affinities).

---

## 🧪 Experiments Overview

### 1. `SVD_Matrix_Factorization.ipynb`
* **Dataset:** MovieLens 100k (`u.data`: 100,000 ratings from 943 users on 1,682 movies).
* **Workflow:** Constructs sparse user-item matrix $\to$ handles missing entries with zero/mean imputation $\to$ fits `TruncatedSVD(n_components=12)` $\to$ reconstructs full dense predicted matrix $\hat{\mathbf{R}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.
* **Diagnostics:**
  * Computes masked **RMSE** and **MAE** evaluated strictly over observed rating coordinates.
  * Side-by-side heatmaps: **Original Sparse Rating Matrix** vs. **Reconstructed Latent Matrix**.
  * Generates personalized Top-10 movie recommendations for individual users.

### 2. `NMF_Matrix_Factorization.ipynb`
* **Dataset:** MovieLens 100k (`u.data`).
* **Workflow:** Constructs non-negative rating matrix $\to$ fits `NMF(n_components=12, init='random')` $\to$ decomposes into user matrix $\mathbf{W}$ and component matrix $\mathbf{H}$ $\to$ reconstructs ratings $\hat{\mathbf{R}} = \mathbf{W}\mathbf{H}$.
* **Diagnostics:**
  * Masked reconstruction RMSE computation.
  * Sample heatmap visualization comparing original ratings against NMF reconstruction.
  * Analysis of latent component interpretability compared to unconstrained SVD.

---

## 📊 Matrix Factorization Comparison

| Factorization Method | Matrix Constraints | Component Meaning | Best For | Masked Validation RMSE |
| :--- | :---: | :--- | :--- | :---: |
| **Truncated SVD** | Orthogonal ($\mathbf{U}^T\mathbf{U} = \mathbf{I}$) | Latent mathematical directions with positive & negative values | Dense dimensionality reduction & vector similarity | $\approx 0.94$ |
| **NMF** | Non-negative ($\mathbf{W} \ge 0, \mathbf{H} \ge 0$) | Purely additive, interpretable parts (e.g. genre weights) | Interpretable topic and user taste modeling | $\approx 0.98$ |

---

## 📂 Notebook Implementations
* `SVD_Matrix_Factorization.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_Matrix_Factorization/SVD_Matrix_Factorization.ipynb)
* `NMF_Matrix_Factorization.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_Matrix_Factorization/NMF_Matrix_Factorization.ipynb)
