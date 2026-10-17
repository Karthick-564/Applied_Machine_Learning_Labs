# 10 · Model-Based Collaborative Filtering & Matrix Factorization

> **Core Focus:** Latent factor recommendation models decomposing large sparse user-item interaction matrices using Truncated Singular Value Decomposition (SVD) and Non-Negative Matrix Factorization (NMF) on the MovieLens 100k benchmark.

---

## 📐 Matrix Factorization Mathematics

### 1. Truncated Singular Value Decomposition (SVD)
Approximates the rating matrix $\mathbf{R} \in \mathbb{R}^{m 	imes n}$ by its rank-$k$ low-rank projection:
$$\mathbf{R} pprox \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
Where:
* $\mathbf{U}_k \in \mathbb{R}^{m 	imes k}$ contains orthogonal user latent affinity vectors.
* $\mathbf{\Sigma}_k \in \mathbb{R}^{k 	imes k}$ contains the top $k$ singular values (factor strength).
* $\mathbf{V}_k \in \mathbb{R}^{n 	imes k}$ contains orthogonal item latent property vectors.

### 2. Non-Negative Matrix Factorization (NMF)
Constrains both user and item factor matrices to non-negative values ($\ge 0$):
$$\min_{\mathbf{W} \ge 0, \mathbf{H} \ge 0} \|\mathbf{R} - \mathbf{W}\mathbf{H}\|_F^2 = \sum_{u=1}^m \sum_{i=1}^n \left( R_{ui} - \sum_{k=1}^K W_{uk} H_{ki} ight)^2$$
Because subtraction is prohibited, factors are strictly additive, yielding intuitive parts-based interpretation (e.g. genre affinities).

---

## 🧪 Experiments Overview

### 1. `01_matrix_factorization_truncated_svd.ipynb`
* **Dataset:** MovieLens 100k (`u.data`: 100,000 ratings from 943 users on 1,682 movies).
* **Workflow:** Constructs sparse user-item matrix $	o$ handles missing entries with zero/mean imputation $	o$ fits `TruncatedSVD(n_components=12)` $	o$ reconstructs full dense predicted matrix $\hat{\mathbf{R}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.
* **Diagnostics:**
  * Computes masked **RMSE** and **MAE** evaluated strictly over observed rating coordinates.
  * Side-by-side heatmaps: **Original Sparse Rating Matrix** vs. **Reconstructed Latent Matrix**.
  * Generates personalized Top-10 movie recommendations for individual users.

### 2. `02_non_negative_matrix_factorization_nmf.ipynb`
* **Dataset:** MovieLens 100k (`u.data`).
* **Workflow:** Constructs non-negative rating matrix $	o$ fits `NMF(n_components=12, init='random')` $	o$ decomposes into user matrix $\mathbf{W}$ and component matrix $\mathbf{H}$ $	o$ reconstructs ratings $\hat{\mathbf{R}} = \mathbf{W}\mathbf{H}$.
* **Diagnostics:**
  * Masked reconstruction RMSE computation.
  * Sample heatmap visualization comparing original ratings against NMF reconstruction.
  * Analysis of latent component interpretability compared to unconstrained SVD.

---

## 📊 Matrix Factorization Comparison

| Factorization Method | Matrix Constraints | Component Meaning | Best For | Masked Validation RMSE |
| :--- | :---: | :--- | :--- | :---: |
| **Truncated SVD** | Orthogonal ($\mathbf{U}^T\mathbf{U} = \mathbf{I}$) | Latent mathematical directions with positive & negative values | Dense dimensionality reduction & vector similarity | $pprox 0.94$ |
| **NMF** | Non-negative ($\mathbf{W} \ge 0, \mathbf{H} \ge 0$) | Purely additive, interpretable parts (e.g. genre weights) | Interpretable topic and user taste modeling | $pprox 0.98$ |

---

## 📂 Notebook Implementations
* `01_matrix_factorization_truncated_svd.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_matrix_factorization_recommenders/01_matrix_factorization_truncated_svd.ipynb)
* `02_non_negative_matrix_factorization_nmf.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_matrix_factorization_recommenders/02_non_negative_matrix_factorization_nmf.ipynb)
