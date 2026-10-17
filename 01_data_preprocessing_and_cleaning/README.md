# 01 · Data Preprocessing & Cleaning Pipelines

> **Core Focus:** Production-grade data sanitation, missing value handling, categorical encoding, feature scaling, and anomaly detection across four real-world commercial and clinical domains.

---

## 🔬 Domain Scenarios & Data Engineering

```
Raw Data Ingestion ───► Anomaly / Null Detection ───► Statistical Imputation ───► Feature Encoding ───► Standardized Output
```

| Scenario | Domain & Dataset | Source Challenges | Applied Transformations | Key Target / Feature Focus |
| :--- | :--- | :--- | :--- | :--- |
| **01 · E-Commerce** | Superstore Sales Transactions | Negative prices, missing sales, invalid order IDs | Median imputation, IQR outlier trimming, MinMax scaling | Sales, Profit, Discount, Quantity |
| **02 · Healthcare** | Clinical Inpatient Records | High-dimensional nulls, dirty demographic text | Mode/Iterative imputation, One-Hot Encoding, StandardScaler | Age, Blood Pressure, Diagnosis Category |
| **03 · Housing** | Real Estate Market Valuations | Highly skewed prices, non-linear spatial attributes | Log transformation $\log(1 + x)$, Target encoding, RobustScaler | SalePrice, Square Footage, ZipCode |
| **04 · Banking** | `marketing_campaign.csv` | Mixed formats, birth years instead of ages | Cyclical datetime parsing, Age extraction, Income outlier capping | Age, Income, Wine & Meat Spending |

---

## 📐 Mathematical Formulations

### 1. Tukey's Interquartile Range (IQR) Outlier Capping
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Valid Domain} = [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$
Any data point $x_i$ outside this interval is flagged as an anomaly and capped to prevent gradient distortion in subsequent downstream estimators.

### 2. Feature Standardization vs. Min-Max Rescaling
* **Z-Score Normalization:**
  $$z = \frac{x - \mu}{\sigma}, \quad \mu = \frac{1}{N}\sum x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum(x_i - \mu)^2}$$
* **Min-Max Normalization (Bounded in $[0, 1]$):**
  $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### 3. Logarithmic De-Skewing for Right-Skewed Distributions
$$y = \log(1 + x)$$
Compresses extreme positive tails, restoring normality for linear models that assume homoscedasticity.

---

## 📂 Notebook Implementations

### 1. `Ecommerce_Cleaning.ipynb`
* **Workflow:** Ingests retail transaction records $\to$ identifies zero and negative values in `Sales` and `Discount` $\to$ imputes missing numerical columns using median values $\to$ caps extreme values with $1.5 \times \text{IQR}$ rule $\to$ visualizes before/after distributions.
* [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Ecommerce_Cleaning.ipynb)

### 2. `Patient_Records.ipynb`
* **Workflow:** Audits patient records for missing laboratory values $\to$ fills categorical missing fields with mode $\to$ converts nominal variables (`Gender`, `Smoker`, `Diagnosis`) into dummy variables via `pd.get_dummies()` $\to$ rescales physiological metrics (`BP`, `Cholesterol`) using `StandardScaler`.
* [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Patient_Records.ipynb)

### 3. `Housing_Transform.ipynb`
* **Workflow:** Analyzes real estate sales $\to$ computes skewness coefficients on `SalePrice` $\to$ applies $\log_e(x)$ transformation $\to$ creates interaction features between square footage and room count $\to$ verifies zero multicollinearity with correlation heatmaps.
* [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Housing_Transform.ipynb)

### 4. `Bank_Marketing.ipynb`
* **Workflow:** Loads customer campaign marketing data (`marketing_campaign.csv`) $\to$ computes current age from `Year_Birth` $\to$ analyzes distribution of `Income` via boxplots $\to$ visualizes customer spending patterns (`Income` vs `MntWines`) via scatter plots $\to$ isolates high-net-worth customer segments.
* [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/01_data_preprocessing_and_cleaning/Bank_Marketing.ipynb)
