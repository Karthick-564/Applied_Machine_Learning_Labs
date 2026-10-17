# 02 · Oceanographic Regression & Stock Classification

> **Core Focus:** Advanced statistical modeling and exploratory diagnostics across two distinct scientific and financial datasets: predicting ocean water temperature using physical oceanography data, and modeling next-day market direction using equity trading records.

---

## 🔬 Detailed Case Studies

### 1. CalCOFI Oceanographic Temperature Regression (`bottle.csv` & `cast.csv`)
* **Scientific Context:** The California Cooperative Oceanic Fisheries Investigations (CalCOFI) dataset represents the longest and most complete oceanographic time series in the world.
* **Objective:** Predict sea surface and subsurface water temperature (`T_degC`) from physical and chemical properties: Salinity (`Salnty`), Depth, and Dissolved Oxygen.
* **Methodology:**
  * Joint merge between station cast metadata and bottle observations.
  * Rigorous handling of ocean depth stratification and missing water samples.
  * Standard scaling of salinity and chemical concentrations.
  * Training **Ordinary Least Squares (OLS) Linear Regression** baseline.
  * Residual error distribution diagnostics ($\hat{y} - y$ vs $\hat{y}$).
  * Optimization and shrinkage comparison with **Ridge ($L_2$)** and **Lasso ($L_1$)** regularization.

### 2. LICI Stock Market Movement Logistic Classification (`LICI_Daily_data.csv`)
* **Financial Context:** Daily trading data of Life Insurance Corporation of India (LICI) equity stock.
* **Objective:** Predict whether next-day closing price will experience a positive or negative price movement based on intraday technical indicators (Open, High, Low, Volume, Daily Return).
* **Methodology:**
  * Target synthesis: Binary indicator $y_t = \mathbb{I}(	ext{Close}_{t+1} > 	ext{Close}_t)$.
  * Feature normalization using `StandardScaler`.
  * Fitting **Logistic Regression** classifier with sigmoid link function:
    $$P(y = 1 \mid \mathbf{x}) = rac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}}$$
  * Comprehensive diagnostic validation: Confusion Matrix, ROC-AUC curve, and L2 hyperparameter tuning.

---

## 📊 Analytical & Evaluation Findings

| Experiment | Dataset | Model Architecture | Key Features | Evaluation Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Ocean Temperature** | CalCOFI (`bottle.csv`) | Linear Regression + Ridge/Lasso | Salinity, Depth, Oxygen | MSE, MAE, $R^2$, Residual Plots |
| **Market Movement** | LICI Stock (`LICI_Daily_data.csv`) | Logistic Regression (L2 tuned) | Open, High, Low, Volume | Accuracy, Precision, Recall, ROC-AUC |

---

## 📂 Notebook Implementations
* `01_oceanographic_temperature_regression.ipynb`: Merging CalCOFI ocean datasets, OLS regression, residual plots, and Ridge/Lasso shrinkage.
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/01_oceanographic_temperature_regression.ipynb)
* `02_stock_movement_logistic_classification.ipynb`: Market direction classification, ROC-AUC curve plotting, confusion matrix, and coefficient importance.
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/02_exploratory_data_analysis/02_stock_movement_logistic_classification.ipynb)
