# 08 · Association Rule Mining & Dimensionality Reduction

> **Core Focus:** Frequent pattern mining and affinity analysis using the Apriori algorithm on retail market baskets, paired with orthogonal covariance eigen-decomposition via Principal Component Analysis (PCA).

---

## 📐 Mathematical Formulation

### 1. Apriori Association Metrics ($X \implies Y$)
* **Support:** Frequency with which itemset appears in transaction database:
  $$\text{Supp}(X \cup Y) = \frac{\sigma(X \cup Y)}{N}$$
* **Confidence:** Conditional probability that transaction containing $X$ also contains $Y$:
  $$\text{Conf}(X \implies Y) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X)}$$
* **Lift:** Ratio of observed joint support to expected support under independence:
  $$\text{Lift}(X \implies Y) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X) \times \text{Supp}(Y)}$$
  *(Lift $> 1$ indicates significant positive association; Lift $< 1$ indicates substitution/negative association).*

### 2. Principal Component Analysis (PCA)
Eigen-decomposition of the standardized feature covariance matrix $\mathbf{\Sigma} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$:
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad \text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$
Orthogonal projection onto the top $k$ eigenvectors transforms high-dimensional features into uncorrelated components capturing maximal variance.

---

## 🧪 Experiments Overview

### 1. `Apriori_Market_Basket.ipynb`
* **Dataset:** Real-World Retail Groceries (`Groceries_dataset.csv`).
* **Methodology:** Aggregates transactions by Member ID and Date $\to$ encodes one-hot transaction matrix via `TransactionEncoder` $\to$ computes frequent itemsets via `apriori(min_support=0.001)` $\to$ mines association rules with high Lift.
* **Diagnostics:**
  * Top 10 frequent itemsets bar chart.
  * **Support vs. Confidence Scatter Plot:** Displays distribution of generated rules colored by Lift.
  * **Network Graph Visualization:** Visualizes directed relationship dependencies between antecedent items and consequent recommendations.

### 2. `PCA_Reduction.ipynb`
* **Dataset:** UCI Wine Recognition Benchmark (13 continuous chemical attributes, 3 cultivar classes).
* **Pipeline:** Standard feature scaling $\to$ `PCA(n_components=None)` $\to$ Scree plot.
* **Diagnostics:**
  * **Scree Plot:** Displays variance captured by individual components.
  * **Cumulative Explained Variance Curve:** Demonstrates that the first 2 principal components capture over **$55\%$** of variance, and 6 components capture $> 85\%$.
  * **2D Projection Scatter (PC1 vs. PC2):** Projects the 13-dimensional wine chemical data onto a 2D plane with distinct separation between wine cultivars.

---

## 📂 Notebook Implementations
* `Apriori_Market_Basket.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/Apriori_Market_Basket.ipynb)
* `PCA_Reduction.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/PCA_Reduction.ipynb)
