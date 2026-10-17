# 09 · Advanced Ensemble Optimization & Hyperparameter Tuning

> **Core Focus:** Multi-tier stacking, meta-classifiers, cross-validated Grid Search and Randomized Search optimization for bias-variance calibration.

---

## ⚙️ Optimization Framework

* **Stacking Architecture:** Base level models (SVM, Random Forest, KNN) generate out-of-fold probability predictions used as inputs for a meta-learner (Logistic Regression).
* **Grid Search Cross-Validation (GridSearchCV):** Exhaustive search across Cartesian product of hyperparameter grids with $K$-fold cross-validation.
* **Evaluation Objectives:** Optimizing $F_1$-score and ROC-AUC while tracking compute time trade-offs.

---

## 📂 Notebook Implementations
* `01_ensemble_voting_and_stacking.ipynb`: Heterogeneous ensemble stacking with blended prediction heads.
* `02_hyperparameter_tuning_gridsearch.ipynb`: Systematic hyperparameter tuning, validation curves, and confusion matrix diagnostics.
