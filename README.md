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
├── 01_data_preprocessing_and_cleaning/     # Real-world retail, clinical, real-estate, and banking cleaning pipelines
├── 02_exploratory_data_analysis/            # CalCOFI oceanographic temperature OLS & LICI stock logistic classification
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
| **01 · Preprocessing** | `01_ecommerce_sales_preprocessing.ipynb` | Superstore Sales | Median imputation, IQR outlier trimming, MinMax scaling | Before/after distribution plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/01_ecommerce_sales_preprocessing.ipynb) |
| **01 · Preprocessing** | `02_healthcare_patient_records.ipynb` | Clinical Records | Mode imputation, One-Hot Encoding, StandardScaler | Standardized laboratory distributions | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/02_healthcare_patient_records.ipynb) |
| **01 · Preprocessing** | `03_housing_feature_transformations.ipynb` | Housing Valuation | Log transformation $\log(1+x)$, RobustScaler, Target encoding | Correlation heatmap, skewness reduction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/03_housing_feature_transformations.ipynb) |
| **01 · Preprocessing** | `04_banking_customer_normalization.ipynb` | `marketing_campaign.csv` | Age extraction, Income boxplot filtering, spending scatter | Income vs. Wine Spending scatter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/04_banking_customer_normalization.ipynb) |
| **02 · Scientific EDA** | `01_oceanographic_temperature_regression.ipynb`| CalCOFI Oceanography | Multiple Linear Regression, Ridge ($L_2$) & Lasso ($L_1$) | Actual vs. Predicted, Residual plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/01_oceanographic_temperature_regression.ipynb) |
| **02 · Financial EDA** | `02_stock_movement_logistic_classification.ipynb`| LICI Stock Daily Data | Logistic Regression with $L_2$ penalty, feature scaling | Confusion Matrix, ROC-AUC curve | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/02_stock_movement_logistic_classification.ipynb) |
| **03 · Regression** | `01_multilinear_regression_student_performance.ipynb`| Student Performance | OLS Multilinear Regression, Ridge, Lasso | Coefficient bar chart, Residual KDE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/01_multilinear_regression_student_performance.ipynb) |
| **03 · Regression** | `02_polynomial_regression_auto_mpg.ipynb`| Auto MPG Benchmark | Polynomial curve fitting ($d \in [2, 4]$), Ridge shrinkage | Continuous curve fit, Train vs. Test MSE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/02_polynomial_regression_auto_mpg.ipynb) |
| **04 · Naïve Bayes** | `01_multinomial_naive_bayes_spam_filter.ipynb`| SMS Spam Collection | Text tokenization, Laplace smoothing ($\alpha = 0.1$), MultinomialNB | Top 10 spam word log-probabilities | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/01_multinomial_naive_bayes_spam_filter.ipynb) |
| **04 · Naïve Bayes** | `02_gaussian_naive_bayes_iris_flowers.ipynb`| Fisher's Iris Dataset | GaussianNB class density modeling, Logistic Regression comparison | 2D Decision boundary contour, KDE plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/02_gaussian_naive_bayes_iris_flowers.ipynb) |
| **05 · Non-Parametric**| `01_k_nearest_neighbors_breast_cancer.ipynb`| Wisconsin Breast Cancer | KNN Euclidean distance classifier, hyperparameter tuning ($K$) | Accuracy vs. $K$ curve, Decision boundary | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_knn_and_decision_trees/01_k_nearest_neighbors_breast_cancer.ipynb) |
| **05 · Tree Models** | `02_decision_tree_loan_prediction.ipynb`| Bank Loan Records | CART Decision Tree, depth regularization (`max_depth`) | Visual tree plot, Feature importance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_knn_and_decision_trees/02_decision_tree_loan_prediction.ipynb) |
| **06 · Ensembles** | `01_bagging_ensemble_diabetes.ipynb` | Pima Indians Diabetes | Bootstrap Aggregation (Bagging) vs. Single Tree | Accuracy gain ($+5.2\%$), Confusion Matrix | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/01_bagging_ensemble_diabetes.ipynb) |
| **06 · Ensembles** | `02_adaboost_and_gradient_boosting_churn.ipynb`| Telco Customer Churn | Sequential Boosting: AdaBoost vs. Gradient Boosting | ROC Curves (AUC: $0.86$), Feature ranking | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/02_adaboost_and_gradient_boosting_churn.ipynb) |
| **06 · Ensembles** | `03_random_forest_income_prediction.ipynb`| Adult Census Income | De-correlated tree bagging with feature subsampling ($m = \sqrt{p}$) | Accuracy vs. Forest size ($B \in [10, 200]$) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/03_random_forest_income_prediction.ipynb) |
| **06 · Ensembles** | `04_stacking_classifier_heart_disease.ipynb`| Heart Disease Clinical | Multi-tier stacking (Logistic Regression + SVM + Tree) | Meta-model accuracy comparison | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/04_stacking_classifier_heart_disease.ipynb) |
| **06 · Ensembles** | `05_smote_class_imbalance_fraud.ipynb` | Credit Card Fraud | SMOTE (Synthetic Minority Over-sampling) + Random Forest | Class distribution plots, PR Curve | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/05_smote_class_imbalance_fraud.ipynb) |
| **07 · Clustering** | `01_kmeans_clustering_elbow_analysis.ipynb`| Mall Customer Spending | Hard geometric partitioning via K-Means | Elbow inertia plot ($K=5$), Silhouette score | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/01_kmeans_clustering_elbow_analysis.ipynb) |
| **07 · Clustering** | `02_gaussian_mixture_models_em.ipynb`| Mall Customer Spending | Soft probabilistic GMM with Expectation-Maximization | BIC model selection curve, Density contours | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/02_gaussian_mixture_models_em.ipynb) |
| **08 · Association**| `01_apriori_market_basket_analysis.ipynb` | Groceries Retail Logs | Apriori frequent itemset mining, Support, Confidence, Lift | Rule network dependency graph | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/01_apriori_market_basket_analysis.ipynb) |
| **08 · Variance PCA**| `02_principal_component_analysis.ipynb` | UCI Wine Recognition | Covariance matrix eigen-decomposition, orthogonal projection | Scree Plot, 2D PC1 vs. PC2 scatter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/02_principal_component_analysis.ipynb) |
| **09 · Recommenders**| `01_user_based_collaborative_filtering.ipynb` | MovieLens Dataset | User-User Cosine Similarity heuristic neighborhood weighting | User-Item heatmap, Masked RMSE/MAE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_memory_based_collaborative_filtering/01_user_based_collaborative_filtering.ipynb) |
| **09 · Recommenders**| `02_item_based_collaborative_filtering.ipynb` | MovieLens Dataset | Item-Item Cosine Similarity heuristic neighborhood weighting | Item similarity matrix, Precision@$K$ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_memory_based_collaborative_filtering/02_item_based_collaborative_filtering.ipynb) |
| **10 · Matrix Factor**| `01_matrix_factorization_truncated_svd.ipynb`| MovieLens 100k (`u.data`) | Low-rank Truncated SVD ($\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$) | Reconstructed matrix heatmap, Masked RMSE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_matrix_factorization_recommenders/01_matrix_factorization_truncated_svd.ipynb) |
| **10 · Matrix Factor**| `02_non_negative_matrix_factorization_nmf.ipynb`| MovieLens 100k (`u.data`) | Non-Negative Matrix Factorization ($\mathbf{W} \ge 0, \mathbf{H} \ge 0$) | Parts-based latent representation heatmap | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_matrix_factorization_recommenders/02_non_negative_matrix_factorization_nmf.ipynb) |

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
