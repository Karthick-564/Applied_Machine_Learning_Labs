# ⚙️ Applied Machine Learning Labs

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Math](https://img.shields.io/badge/Math-NumPy%20%7C%20SciPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Data](https://img.shields.io/badge/Data-Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-232F3E?style=flat)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A rigorous, production-grade machine learning laboratory suite spanning 10 core modules and 25 notebooks. Implemented across genuine scientific, retail, financial, and clinical datasets: Data Sanitation Pipelines, Oceanographic & Stock Modeling, Polynomial Complexity Diagnostics, Probabilistic Generative Models, Margin Classifiers, Tree Ensembles with SMOTE, Hard vs. Soft Clustering, Association Rule Mining, and Matrix Factorization Recommender Systems.

---

## 🗺️ Laboratory Curriculum & Module Map

```text
applied-machine-learning-labs/
├── 01_data_preprocessing_and_cleaning/     # Retail, clinical, housing, and banking cleaning pipelines
├── 02_exploratory_data_analysis/            # CalCOFI oceanographic temperature regression & LICI stock classification
├── 03_regression_models/                    # Student performance multilinear OLS & Auto MPG polynomial curve fitting
├── 04_naive_bayes_classification/           # SMS spam detection (Multinomial NB) & Iris morphology (Gaussian NB)
├── 05_knn_and_decision_trees/               # Breast cancer KNN decision boundaries & Loan approval decision trees
├── 06_tree_based_ensembles/                 # Bagging, AdaBoost, Gradient Boosting, Random Forests, Stacking & SMOTE
├── 07_unsupervised_clustering/              # Mall customer segmentation via K-Means (Elbow/Silhouette) & GMM (EM)
├── 08_association_rules_and_pca/            # Apriori retail market basket analysis & UCI Wine covariance PCA decomposition
├── 09_memory_based_collaborative_filtering/ # MovieLens User-User and Item-Item heuristic collaborative filtering
└── 10_matrix_factorization_recommenders/    # MovieLens 100k latent factor decomposition via Truncated SVD and NMF
```

---

## 🔬 Master Experiment Index (25 Notebooks)

| Module | Notebook | Real-World Dataset | Key Algorithms & Methods | Diagnostic Output & Metrics | Interactive Colab |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **01 · Preprocessing** | [`Ecommerce_Cleaning.ipynb`](./01_data_preprocessing_and_cleaning/Ecommerce_Cleaning.ipynb) | Superstore Sales | Median imputation, IQR outlier trimming, MinMax scaling | Before/after distribution plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Ecommerce_Cleaning.ipynb) |
| **01 · Preprocessing** | [`Patient_Records.ipynb`](./01_data_preprocessing_and_cleaning/Patient_Records.ipynb) | Clinical Records | Mode imputation, One-Hot Encoding, StandardScaler | Standardized laboratory distributions | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Patient_Records.ipynb) |
| **01 · Preprocessing** | [`Housing_Transform.ipynb`](./01_data_preprocessing_and_cleaning/Housing_Transform.ipynb) | Housing Valuation | Log transformation $\log(1+x)$, RobustScaler, Target encoding | Correlation heatmap, skewness reduction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Housing_Transform.ipynb) |
| **01 · Preprocessing** | [`Bank_Marketing.ipynb`](./01_data_preprocessing_and_cleaning/Bank_Marketing.ipynb) | `marketing_campaign.csv` | Age extraction, Income boxplot filtering, spending scatter | Income vs. Wine Spending scatter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Bank_Marketing.ipynb) |
| **02 · Scientific EDA** | [`Ocean_Regression.ipynb`](./02_exploratory_data_analysis/Ocean_Regression.ipynb) | CalCOFI Oceanography | Multiple Linear Regression, Ridge ($L_2$) & Lasso ($L_1$) | Actual vs. Predicted, Residual plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/Ocean_Regression.ipynb) |
| **02 · Financial EDA** | [`Stock_Classification.ipynb`](./02_exploratory_data_analysis/Stock_Classification.ipynb) | LICI Stock Daily Data | Logistic Regression with $L_2$ penalty, feature scaling | Confusion Matrix, ROC-AUC curve | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/Stock_Classification.ipynb) |
| **03 · Regression** | [`Linear_Regression.ipynb`](./03_regression_models/Linear_Regression.ipynb) | Student Performance | OLS Multilinear Regression, Ridge, Lasso | Coefficient bar chart, Residual KDE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/Linear_Regression.ipynb) |
| **03 · Regression** | [`Polynomial_Regression.ipynb`](./03_regression_models/Polynomial_Regression.ipynb) | Auto MPG Benchmark | Polynomial curve fitting ($d \in [2, 4]$), Ridge shrinkage | Continuous curve fit, Train vs. Test MSE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/Polynomial_Regression.ipynb) |
| **04 · Naïve Bayes** | [`Spam_Filter_NB.ipynb`](./04_naive_bayes_classification/Spam_Filter_NB.ipynb) | SMS Spam Collection | Text tokenization, Laplace smoothing ($\alpha = 0.1$), MultinomialNB | Top 10 spam word log-probabilities | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/Spam_Filter_NB.ipynb) |
| **04 · Naïve Bayes** | [`Iris_Gaussian_NB.ipynb`](./04_naive_bayes_classification/Iris_Gaussian_NB.ipynb) | Fisher's Iris Dataset | GaussianNB class density modeling, Logistic Regression comparison | 2D Decision boundary contour, KDE plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/Iris_Gaussian_NB.ipynb) |
| **05 · Non-Parametric**| [`KNN_Cancer.ipynb`](./05_knn_and_decision_trees/KNN_Cancer.ipynb) | Wisconsin Breast Cancer | KNN Euclidean distance classifier, hyperparameter tuning ($K$) | Accuracy vs. $K$ curve, Decision boundary | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_knn_and_decision_trees/KNN_Cancer.ipynb) |
| **05 · Tree Models** | [`Decision_Tree_Loan.ipynb`](./05_knn_and_decision_trees/Decision_Tree_Loan.ipynb) | Bank Loan Records | CART Decision Tree, depth regularization (`max_depth`) | Visual tree plot, Feature importance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_knn_and_decision_trees/Decision_Tree_Loan.ipynb) |
| **06 · Ensembles** | [`Bagging_Diabetes.ipynb`](./06_tree_based_ensembles/Bagging_Diabetes.ipynb) | Pima Indians Diabetes | Bootstrap Aggregation (Bagging) vs. Single Tree | Accuracy gain ($+5.2\%$), Confusion Matrix | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Bagging_Diabetes.ipynb) |
| **06 · Ensembles** | [`Boosting_Churn.ipynb`](./06_tree_based_ensembles/Boosting_Churn.ipynb) | Telco Customer Churn | Sequential Boosting: AdaBoost vs. Gradient Boosting | ROC Curves (AUC: $0.86$), Feature ranking | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Boosting_Churn.ipynb) |
| **06 · Ensembles** | [`Random_Forest_Income.ipynb`](./06_tree_based_ensembles/Random_Forest_Income.ipynb) | Adult Census Income | De-correlated tree bagging with feature subsampling ($m = \sqrt{p}$) | Accuracy vs. Forest size ($B \in [10, 200]$) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Random_Forest_Income.ipynb) |
| **06 · Ensembles** | [`Stacking_Heart.ipynb`](./06_tree_based_ensembles/Stacking_Heart.ipynb) | Heart Disease Clinical | Multi-tier stacking (Logistic Regression + SVM + Tree) | Meta-model accuracy comparison | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/Stacking_Heart.ipynb) |
| **06 · Ensembles** | [`SMOTE_Fraud.ipynb`](./06_tree_based_ensembles/SMOTE_Fraud.ipynb) | Credit Card Fraud | SMOTE (Synthetic Minority Over-sampling) + Random Forest | Class distribution plots, PR Curve | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/SMOTE_Fraud.ipynb) |
| **07 · Clustering** | [`KMeans_Clustering.ipynb`](./07_unsupervised_clustering/KMeans_Clustering.ipynb) | Mall Customer Spending | Hard geometric partitioning via K-Means | Elbow inertia plot ($K=5$), Silhouette score | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/KMeans_Clustering.ipynb) |
| **07 · Clustering** | [`GMM_Clustering.ipynb`](./07_unsupervised_clustering/GMM_Clustering.ipynb) | Mall Customer Spending | Soft probabilistic GMM with Expectation-Maximization | BIC model selection curve, Density contours | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/GMM_Clustering.ipynb) |
| **08 · Association**| [`Apriori_Market_Basket.ipynb`](./08_association_rules_and_pca/Apriori_Market_Basket.ipynb) | Groceries Retail Logs | Apriori frequent itemset mining, Support, Confidence, Lift | Rule network dependency graph | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/Apriori_Market_Basket.ipynb) |
| **08 · Variance PCA**| [`PCA_Reduction.ipynb`](./08_association_rules_and_pca/PCA_Reduction.ipynb) | UCI Wine Recognition | Covariance matrix eigen-decomposition, orthogonal projection | Scree Plot, 2D PC1 vs. PC2 scatter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/PCA_Reduction.ipynb) |
| **09 · Recommenders**| [`User_Collaborative_Filtering.ipynb`](./09_memory_based_collaborative_filtering/User_Collaborative_Filtering.ipynb) | MovieLens Dataset | User-User Cosine Similarity heuristic neighborhood weighting | User-Item heatmap, Masked RMSE/MAE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_memory_based_collaborative_filtering/User_Collaborative_Filtering.ipynb) |
| **09 · Recommenders**| [`Item_Collaborative_Filtering.ipynb`](./09_memory_based_collaborative_filtering/Item_Collaborative_Filtering.ipynb) | MovieLens Dataset | Item-Item Cosine Similarity heuristic neighborhood weighting | Item similarity matrix, Precision@$K$ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_memory_based_collaborative_filtering/Item_Collaborative_Filtering.ipynb) |
| **10 · Matrix Factor**| [`SVD_Matrix_Factorization.ipynb`](./10_matrix_factorization_recommenders/SVD_Matrix_Factorization.ipynb) | MovieLens 100k (`u.data`) | Low-rank Truncated SVD ($\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$) | Reconstructed matrix heatmap, Masked RMSE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_matrix_factorization_recommenders/SVD_Matrix_Factorization.ipynb) |
| **10 · Matrix Factor**| [`NMF_Matrix_Factorization.ipynb`](./10_matrix_factorization_recommenders/NMF_Matrix_Factorization.ipynb) | MovieLens 100k (`u.data`) | Non-Negative Matrix Factorization ($\mathbf{W} \ge 0, \mathbf{H} \ge 0$) | Parts-based latent representation heatmap | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_matrix_factorization_recommenders/NMF_Matrix_Factorization.ipynb) |

---

## 📊 Core Theoretical & Empirical Insights

### 1. Model Complexity & Bias-Variance Tradeoff
* **Polynomial Regression on Auto MPG:** Testing degrees $d \in [1, 4]$ proves that training loss decreases monotonically with degree, but test error rebounds sharply at $d = 4$ due to wild boundary oscillations. Introducing **Ridge $L_2$ regularization** ($\lambda = 10$) contracts the parameter vector, suppressing catastrophic variance.
* **K-Nearest Neighbors on Breast Cancer:** Evaluating $K \in [1, 25]$ demonstrates that $K=1$ forms an overfitted Voronoi tessellation sensitive to label noise, whereas $K \approx 7-9$ provides the optimal bias-variance equilibrium.

### 2. The Power of Ensemble Diversity
* **Bagging vs. Single Trees (Diabetes):** Averaging bootstrap estimators reduces prediction variance without inflating bias, lifting accuracy from $71.4\%$ to **$77.9\%$**.
* **Boosting (Telco Churn):** Gradient Boosting constructs an additive sequence of shallow trees fitting pseudo-residuals, outperforming AdaBoost with a superior ROC-AUC of **$0.86$**.
* **Class Imbalance Remediation (Credit Fraud):** In severe imbalance ($< 1\%$ positive minority), traditional models degenerate into predicting the majority class. **SMOTE** synthetic interpolation restores balance, boosting minority recall from $0.34$ to **$0.88$**.

### 3. Clustering Mechanics: Geometry vs. Probability
* **K-Means vs. GMM (Mall Customers):** While K-Means enforces spherical, equidistant Euclidean partitioning around centroids, Gaussian Mixture Models capture directional covariance and assign probabilistic membership degrees, effectively handling overlapping, non-spherical customer cohorts.

### 4. Recommendation Dynamics: Heuristic vs. Latent Space
* **Memory-Based (Collaborative Filtering):** Direct Cosine Similarity neighbor search is intuitive and explainable but struggles with extreme matrix sparsity.
* **Model-Based (Truncated SVD & NMF):** Dimensionality reduction projects sparse user-item interactions into dense rank-$k$ latent semantic factor vectors, achieving superior generalization and imputation accuracy on unobserved coordinates.

---

## 🚀 Quickstart & Reproduction

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

```bash
# 1. Clone the repository
git clone https://github.com/Karthick-564/applied-machine-learning-labs.git
cd applied-machine-learning-labs

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch interactive Jupyter Lab
jupyter lab
```

Or execute any experiment directly in **Google Colab** by clicking the badges in the [Master Experiment Index](#-master-experiment-index-25-notebooks).

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
