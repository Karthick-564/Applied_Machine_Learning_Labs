# ⚙️ Applied Machine Learning Labs

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Math](https://img.shields.io/badge/Math-NumPy%20%7C%20SciPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Data](https://img.shields.io/badge/Data-Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-232F3E?style=flat)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive, production-oriented machine learning laboratory suite covering end-to-end data pipelines: Data Preprocessing, Exploratory Data Analysis, Linear & Polynomial Regression, Naïve Bayes, Support Vector Machines, Tree-Based Ensembles, Unsupervised Clustering, Association Rule Mining, Dimensionality Reduction (PCA), and Collaborative Filtering Recommendation Systems.

---

## 🗺️ Curriculum & Laboratory Modules

```text
applied-machine-learning-labs/
├── 01_data_preprocessing_and_cleaning/     # Missing values, IQR clipping, scaling across 4 business domains
├── 02_exploratory_data_analysis/            # Multivariate distributions, boxplots, and correlation heatmaps
├── 03_regression_models/                    # OLS, Batch Gradient Descent, and polynomial curve fitting
├── 04_naive_bayes_classification/           # Text tokenization, Laplace smoothing, and multinomial NLP
├── 05_support_vector_machines/              # Maximum margin hyperplanes and RBF non-linear kernels
├── 06_tree_based_ensembles/                 # Decision Trees, Random Forests, AdaBoost, and ROC-AUC benchmarks
├── 07_unsupervised_clustering/              # K-Means (Elbow & Silhouette) and Gaussian Mixture Models (EM)
├── 08_association_rules_and_pca/            # Apriori Market Basket Analysis and PCA variance decomposition
├── 09_ensemble_optimization_and_tuning/     # Stacking meta-learners and cross-validated GridSearchCV
└── 10_recommendation_systems/               # Collaborative filtering via Truncated SVD and NMF
```

---

## 🔬 Master Experiment Index

| # | Module | Notebook | Core Algorithms & Concepts | Colab |
| :-: | :--- | :--- | :--- | :-: |
| **01** | Data Preprocessing | `01_ecommerce_sales_preprocessing.ipynb` | Retail transaction cleaning, outlier filtering, MinMax scaling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/01_ecommerce_sales_preprocessing.ipynb) |
| **01** | Data Preprocessing | `02_healthcare_patient_records.ipynb` | Clinical null imputation, one-hot categorical encoding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/02_healthcare_patient_records.ipynb) |
| **01** | Data Preprocessing | `03_housing_feature_transformations.ipynb` | Log transformations $\log(1+x)$, RobustScaler, spatial features | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/03_housing_feature_transformations.ipynb) |
| **01** | Data Preprocessing | `04_banking_customer_normalization.ipynb` | Cyclical datetime parsing, financial risk Z-score normalization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/04_banking_customer_normalization.ipynb) |
| **02** | EDA & Statistics | `01_multivariate_distributions_boxplots.ipynb` | KDE probability profiling, skewness, and group-wise spread | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/01_multivariate_distributions_boxplots.ipynb) |
| **02** | EDA & Statistics | `02_correlation_and_outlier_detection.ipynb` | Pearson $r$, Spearman $\rho$, and Tukey's Fences IQR isolation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/02_correlation_and_outlier_detection.ipynb) |
| **03** | Regression | `01_linear_regression_gradient_descent.ipynb` | Manual Gradient Descent optimization vs. Scikit-Learn OLS | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/01_linear_regression_gradient_descent.ipynb) |
| **03** | Regression | `02_polynomial_regression_curve_fitting.ipynb` | Polynomial feature synthesis ($d \in [1, 5]$) and bias-variance tradeoff | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/03_regression_models/02_polynomial_regression_curve_fitting.ipynb) |
| **04** | Naïve Bayes | `01_text_preprocessing_and_tokenization.ipynb` | NLP text cleansing, stopword removal, and vocabulary mapping | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/01_text_preprocessing_and_tokenization.ipynb) |
| **04** | Naïve Bayes | `02_multinomial_naive_bayes_classifier.ipynb` | Multinomial Bayes with Laplace smoothing ($\alpha = 1.0$) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/02_multinomial_naive_bayes_classifier.ipynb) |
| **05** | Support Vector Machines | `01_linear_svm_margin_maximization.ipynb` | Maximum margin separation and penalty factor $C$ calibration | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_support_vector_machines/01_linear_svm_margin_maximization.ipynb) |
| **05** | Support Vector Machines | `02_kernel_svm_rbf_nonlinear_boundaries.ipynb` | Non-linear RBF kernel transformation and parameter $\gamma$ tuning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/05_support_vector_machines/02_kernel_svm_rbf_nonlinear_boundaries.ipynb) |
| **06** | Tree Ensembles | `01_decision_trees_and_pruning.ipynb` | Gini impurity splitting and cost-complexity pruning (`ccp_alpha`) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/01_decision_trees_and_pruning.ipynb) |
| **06** | Tree Ensembles | `02_logistic_regression_baseline.ipynb` | Sigmoid odds-ratio estimation and linear decision thresholding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/02_logistic_regression_baseline.ipynb) |
| **06** | Tree Ensembles | `03_random_forests_feature_importance.ipynb` | Bootstrap aggregation (Bagging) and MDI feature importance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/03_random_forests_feature_importance.ipynb) |
| **06** | Tree Ensembles | `04_ensemble_voting_and_bagging.ipynb` | Hard vs. Soft probability voting across diverse learners | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/04_ensemble_voting_and_bagging.ipynb) |
| **06** | Tree Ensembles | `05_model_comparative_benchmark.ipynb` | Comprehensive ROC-AUC, Precision, Recall, and $F_1$ evaluation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/06_tree_based_ensembles/05_model_comparative_benchmark.ipynb) |
| **07** | Clustering | `01_kmeans_clustering_elbow_analysis.ipynb` | WCSS inertia minimization and Silhouette score validation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/01_kmeans_clustering_elbow_analysis.ipynb) |
| **07** | Clustering | `02_gaussian_mixture_models_em.ipynb` | Expectation-Maximization probabilistic soft cluster assignment | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/07_unsupervised_clustering/02_gaussian_mixture_models_em.ipynb) |
| **08** | Association & PCA | `01_apriori_market_basket_analysis.ipynb` | Frequent itemsets, Support, Confidence, and Lift metrics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/01_apriori_market_basket_analysis.ipynb) |
| **08** | Association & PCA | `02_principal_component_analysis.ipynb` | Covariance eigen-decomposition and 2D spatial variance reduction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/08_association_rules_and_pca/02_principal_component_analysis.ipynb) |
| **09** | Ensembles & Tuning | `01_ensemble_voting_and_stacking.ipynb` | Multi-layer stacking with Logistic Regression meta-classifier | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_ensemble_optimization_and_tuning/01_ensemble_voting_and_stacking.ipynb) |
| **09** | Ensembles & Tuning | `02_hyperparameter_tuning_gridsearch.ipynb` | $K$-fold GridSearchCV hyperparameter calibration | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/09_ensemble_optimization_and_tuning/02_hyperparameter_tuning_gridsearch.ipynb) |
| **10** | Recommender Systems | `01_collaborative_filtering_truncated_svd.ipynb` | Rating matrix decomposition $\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_recommendation_systems/01_collaborative_filtering_truncated_svd.ipynb) |
| **10** | Recommender Systems | `02_non_negative_matrix_factorization.ipynb` | Non-negative parts-based matrix factorization ($\mathbf{W} \ge 0, \mathbf{H} \ge 0$) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/10_recommendation_systems/02_non_negative_matrix_factorization.ipynb) |

---

## 🚀 Quickstart & Reproduction

### Prerequisites
* Python 3.10+
* Virtual environment (`venv` or `conda`)

```bash
# 1. Clone repository
git clone https://github.com/Karthick-564/applied-machine-learning-labs.git
cd applied-machine-learning-labs

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter lab
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
