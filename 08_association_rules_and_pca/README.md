# 08 · Association Rule Mining & Dimensionality Reduction

> **Core Focus:** Frequent pattern mining and affinity analysis using the Apriori algorithm on retail market baskets, paired with orthogonal covariance eigen-decomposition via Principal Component Analysis (PCA).

---

## 📐 Mathematical Formulation

### 1. Apriori Association Metrics ($X \implies Y$)
* **Support:** Frequency with which itemset appears in transaction database:
  $$	ext{Supp}(X \cup Y) = rac{\sigma(X \cup Y)}{N}$$
* **Confidence:** Conditional probability that transaction containing $X$ also contains $Y$:
  $$	ext{Conf}(X \implies Y) = rac{	ext{Supp}(X \cup Y)}{	ext{Supp}(X)}$$
* **Lift:** Ratio of observed joint support to expected support under independence:
  $$	ext{Lift}(X \implies Y) = rac{	ext{Supp}(X \cup Y)}{	ext{Supp}(X) 	imes 	ext{Supp}(Y)}$$
  *(Lift $> 1$ indicates significant positive association; Lift $< 1$ indicates substitution/negative association).*

### 2. Principal Component Analysis (PCA)
Eigen-decomposition of the standardized feature covariance matrix $\mathbf{\Sigma} = rac{1}{N} \mathbf{X}^T \mathbf{X}$:
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad 	ext{Explained Variance Ratio} = rac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$
Orthogonal projection onto the top $k$ eigenvectors transforms high-dimensional features into uncorrelated components capturing maximal variance.

---

## 🧪 Experiments Overview

### 1. `01_apriori_market_basket_analysis.ipynb`
* **Dataset:** Real-World Retail Groceries (`Groceries_dataset.csv`).
* **Methodology:** Aggregates transactions by Member ID and Date $	o$ encodes one-hot transaction matrix via `TransactionEncoder` $	o$ computes frequent itemsets via `apriori(min_support=0.001)` $	o$ mines association rules with high Lift.
* **Diagnostics:**
  * Top 10 frequent itemsets bar chart.
  * **Support vs. Confidence Scatter Plot:** Displays distribution of generated rules colored by Lift.
  * **Network Graph Visualization:** Visualizes directed relationship dependencies between antecedent items and consequent recommendations.

### 2. `02_principal_component_analysis.ipynb`
* **Dataset:** UCI Wine Recognition Benchmark (13 continuous chemical attributes, 3 cultivar classes).
* **Pipeline:** Standard feature scaling $	o$ `PCA(n_components=None)` $	o$ Scree plot.
* **Diagnostics:**
  * **Scree Plot:** Displays variance captured by individual components.
  * **Cumulative Explained Variance Curve:** Demonstrates that the first 2 principal components capture over **$55\%$** of variance, and 6 components capture $> 85\%$.
  * **2D Projection Scatter (PC1 vs. PC2):** Projects the 13-dimensional wine chemical data onto a 2D plane with distinct separation between wine cultivars.

---

## 📂 Notebook Implementations
* `01_apriori_market_basket_analysis.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/01_apriori_market_basket_analysis.ipynb)
* `02_principal_component_analysis.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/02_principal_component_analysis.ipynb)
