# 08 · Association Rule Mining & Dimensionality Reduction

> **Core Focus:** Frequent itemset mining via Apriori algorithm, market basket analysis metrics (Support, Confidence, Lift), and orthogonal variance decomposition via Principal Component Analysis (PCA).

---

## 📐 Formulations

### 1. Association Rule Metrics: $X \implies Y$
* **Support:** $\text{Supp}(X \cup Y) = \frac{\sigma(X \cup Y)}{N}$
* **Confidence:** $\text{Conf}(X \implies Y) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X)}$
* **Lift:** $\text{Lift}(X \implies Y) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X) \times \text{Supp}(Y)}$ (Lift $> 1$ indicates significant positive association).

### 2. Principal Component Analysis (PCA)
Eigen-decomposition of the centered feature covariance matrix $\mathbf{\Sigma} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$:
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad \text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

---

## 📂 Notebook Implementations
* `01_apriori_market_basket_analysis.ipynb`: Transaction encoding, frequent pattern mining, and association rule filtering based on Lift and Confidence thresholds.
* `02_principal_component_analysis.ipynb`: Scree plot visualization, cumulative explained variance analysis, and dimensionality reduction down to 2 principal components.
