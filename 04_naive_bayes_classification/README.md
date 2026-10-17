# 04 · Naïve Bayes Classification (Multinomial & Gaussian)

> **Core Focus:** Probabilistic generative classification, Bayes' Theorem with conditional feature independence, text frequency modeling with Laplace smoothing, and continuous feature density modeling via Gaussian Naïve Bayes.

---

## 📐 Mathematical Framework

### 1. Bayes' Rule with Conditional Independence
$$P(y = c \mid \mathbf{x}) = rac{P(y = c) \prod_{j=1}^d P(x_j \mid y = c)}{P(\mathbf{x})}$$

### 2. Multinomial Model for Text (with Laplace Smoothing)
$$P(w_j \mid y = c) = rac{N_{cj} + lpha}{N_c + lpha |V|}$$
Where $lpha = 0.1$ prevents zero probability penalties for rare words absent from specific training classes.

### 3. Gaussian Model for Continuous Variables
$$P(x_j \mid y = c) = rac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left( -rac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} ight)$$

---

## 🧪 Experiments Overview

### 1. `01_multinomial_naive_bayes_spam_filter.ipynb`
* **Task:** SMS Spam Detection (Spam vs. Ham).
* **Pipeline:** Raw SMS text $	o$ punctuation removal & lowercasing $	o$ word tokenization $	o$ Term Frequency vectorization $	o$ `MultinomialNB(alpha=0.1)`.
* **Diagnostics:**
  * Confusion matrix heatmap showing true positive vs. false positive spam rates.
  * **Top 10 Influential Spam Words:** Horizontal bar chart of log-probability features indicating strongest spam signals (e.g. *free*, *claim*, *call*, *win*).
  * **Word Frequency Comparison:** Frequency counts comparing distribution between Spam and Ham corpora.

### 2. `02_gaussian_naive_bayes_iris_flowers.ipynb`
* **Task:** Multi-class classification on the canonical Fisher's Iris dataset (Setosa, Versicolor, Virginica).
* **Pipeline:** 4 continuous morphological features $	o$ Standard scaling $	o$ `GaussianNB()`.
* **Diagnostics:**
  * Macro-averaged Precision, Recall, and $F_1$-score.
  * **Probability Density Plots (KDE):** Class-conditional Gaussian distributions $P(	ext{Petal Length} \mid 	ext{Species})$.
  * **2D Decision Boundary:** Meshgrid contour plot showing non-linear elliptical decision boundaries separating the three flower species.
  * Direct comparison with Logistic Regression.

---

## 📂 Notebook Implementations
* `01_multinomial_naive_bayes_spam_filter.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/01_multinomial_naive_bayes_spam_filter.ipynb)
* `02_gaussian_naive_bayes_iris_flowers.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/02_gaussian_naive_bayes_iris_flowers.ipynb)
