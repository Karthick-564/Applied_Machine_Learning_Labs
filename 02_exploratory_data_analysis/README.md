# 02 · Exploratory Data Analysis (EDA) & Statistical Inference

> **Core Focus:** Univariate and bivariate probability distributions, parametric vs. non-parametric hypothesis testing, Pearson & Spearman correlation structures, and boxplot anomaly isolation.

---

## 📊 Analytical Methodology

* **Univariate Profiling:** Kernel Density Estimation (KDE), histogram binning analysis, skewness, and kurtosis estimation.
* **Multivariate Association:** Pairwise Pearson correlation matrices ($r$) for linear relationships and Spearman rank correlation ($\rho$) for monotonic dependencies:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
* **Outlier Boundary Detection:** Tukey's Fences method based on Interquartile Range (IQR):
  $$\text{Fences} = [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$

---

## 📂 Notebook Implementations
* `01_multivariate_distributions_boxplots.ipynb`: Statistical distribution testing, visual normality checks, and feature spread across groups.
* `02_correlation_and_outlier_detection.ipynb`: Correlation heatmaps, collinearity diagnostics, and automated Tukey IQR filtering.
