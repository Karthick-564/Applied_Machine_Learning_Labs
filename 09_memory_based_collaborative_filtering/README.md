# 09 · Memory-Based Collaborative Filtering Recommenders

> **Core Focus:** Nearest-neighbor heuristic recommendation systems implemented on the MovieLens dataset, covering both User-Based Collaborative Filtering (UBCF) and Item-Based Collaborative Filtering (IBCF) with Cosine Similarity weighting.

---

## 📐 Heuristic Recommendation Formulations

### 1. Cosine Similarity Function
Measures directional alignment between user or item rating vectors in sparse vector space:
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i \in I_{uv}} r_{ui} r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}$$

### 2. User-Based Rating Prediction (UBCF)
Predicts rating of active user $u$ for item $i$ via similarity-weighted neighborhood average:
$$\hat{r}_{u, i} = \frac{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v) \cdot r_{v, i}}{\sum_{v \in \mathcal{N}_k(u)} |\text{sim}(u, v)|}$$

### 3. Item-Based Rating Prediction (IBCF)
Predicts user $u$'s rating for item $i$ based on ratings user $u$ gave to similar items $j$:
$$\hat{r}_{u, i} = \frac{\sum_{j \in \mathcal{S}_k(i)} \text{sim}(i, j) \cdot r_{u, j}}{\sum_{j \in \mathcal{S}_k(i)} |\text{sim}(i, j)|}$$

---

## 🧪 Experiments Overview

### 1. `User_Collaborative_Filtering.ipynb`
* **Dataset:** MovieLens (`ratings.csv` & `movies.csv`).
* **Workflow:** Pivots rating logs into $U \times I$ sparse matrix $\to$ computes pairwise User Cosine Similarity matrix $\to$ identifies top-$N$ most similar peer users $\to$ generates top personalized unrated movie recommendations.
* **Diagnostics:**
  * **User-Item Matrix Heatmap:** Visualizes rating sparsity and dense rating clusters.
  * **User Similarity Heatmap:** Displays correlation clustering among user subgroups.
  * Evaluates overall prediction error using masked **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

### 2. `Item_Collaborative_Filtering.ipynb`
* **Dataset:** MovieLens (`ratings.csv` & `movies.csv`).
* **Workflow:** Transposes into Item-User matrix $\to$ computes Item-Item Cosine Similarity matrix $\to$ queries top similar movies for target films (e.g. *Star Wars*, *Toy Story*) $\to$ predicts user preferences.
* **Diagnostics:**
  * **Item Similarity Heatmap:** Matrix visualization of inter-film similarity.
  * **Bar Graph:** Top-5 most similar titles for given query movies.
  * **Head-to-Head Comparison:** Evaluates User-Based vs. Item-Based recommendation accuracy and computes **Precision@$K$**.

---

## 📂 Notebook Implementations
* `User_Collaborative_Filtering.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_memory_based_collaborative_filtering/User_Collaborative_Filtering.ipynb)
* `Item_Collaborative_Filtering.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_memory_based_collaborative_filtering/Item_Collaborative_Filtering.ipynb)
