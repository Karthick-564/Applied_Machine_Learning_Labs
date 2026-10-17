# 06 · Tree-Based Models & Ensemble Architecture

> **Core Focus:** Recursive binary splitting, impurity measures (Gini & Entropy), cost-complexity pruning, Random Forests, and AdaBoost / Gradient Boosting.

---

## 🌲 Splitting Criteria & Impurity

### Gini Impurity & Shannon Entropy
$$I_G(p) = 1 - \sum_{k=1}^K p_k^2, \quad H(p) = -\sum_{k=1}^K p_k \log_2(p_k)$$

### Random Forest Variance Reduction
Averages $B$ de-correlated trees trained on bootstrap samples with random feature subsets $m \approx \sqrt{p}$:
$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

---

## 📂 Notebook Implementations
* `01_decision_trees_and_pruning.ipynb`: Recursive tree generation and pre/post-pruning via `max_depth` and `ccp_alpha`.
* `02_logistic_regression_baseline.ipynb`: Linear probabilistic baseline benchmark.
* `03_random_forests_feature_importance.ipynb`: Bagging ensemble implementation and Mean Decrease in Impurity (MDI) feature ranking.
* `04_ensemble_voting_and_bagging.ipynb`: Soft vs. hard voting aggregation across diverse base learners.
* `05_model_comparative_benchmark.ipynb`: Head-to-head ROC-AUC, Precision, Recall, and $F_1$ evaluation across all model variants.
