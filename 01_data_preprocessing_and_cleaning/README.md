# 01 · Data Preprocessing & Cleaning Pipelines

> **Core Focus:** End-to-end data sanitation, missing value handling, categorical encoding, feature scaling, and anomaly detection across four real-world business scenarios.

---

## 🔬 Scenario Overview & Data Engineering

| Scenario | Domain | Challenges Addressed | Core Techniques |
| :--- | :--- | :--- | :--- |
| **01 · E-Commerce Sales** | Retail Analytics | Missing price/quantity values, extreme transaction outliers | Median imputation, IQR clipping, MinMax normalization |
| **02 · Healthcare Records** | Clinical Patient Logs | High-dimensional nulls, dirty demographic text | Mode/Iterative imputation, One-Hot Encoding, domain constraints |
| **03 · Housing Transactions**| Real Estate | Skewed property values, high-cardinality zip codes | Log transformation $\log(1 + x)$, Target encoding, RobustScaler |
| **04 · Banking Customers** | Financial Risk | Unformatted timestamps, mixed currency scales | Epoch/cyclical datetime extraction, Z-score standardisation |

---

## 📐 Mathematical Foundations

### 1. Robust Z-Score Normalization
$$z = \frac{x - \mu}{\sigma}, \quad z_{\text{robust}} = \frac{x - \text{median}}{\text{IQR}}$$

### 2. Min-Max Rescaling to $[0, 1]$
$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### 3. Logarithmic De-Skewing
$$y = \log(1 + x), \quad \text{reduces positive skewness in long-tailed financial variables.}$$

---

## 📂 Notebook Implementations
* `01_ecommerce_sales_preprocessing.ipynb`: Transaction cleaning & purchase anomaly detection.
* `02_healthcare_patient_records.ipynb`: Patient demographic & diagnostic record sanitation.
* `03_housing_feature_transformations.ipynb`: Spatial and numeric feature engineering for valuation.
* `04_banking_customer_normalization.ipynb`: Temporal parsing and customer risk profile standardization.
