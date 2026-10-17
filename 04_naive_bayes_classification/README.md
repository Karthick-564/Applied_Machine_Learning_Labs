# 04 · Naïve Bayes Classification (Multinomial & Gaussian)

> **Core Focus:** Probabilistic generative classification, Bayes' Theorem with conditional feature independence, text frequency modeling with Laplace smoothing, and continuous feature density modeling via Gaussian Naïve Bayes.

---

## 📐 Mathematical Framework

### 1. Bayes' Rule with Conditional Independence
$$P(y = c \mid \mathbf{x}) = \frac{P(y = c) \prod_{j=1}^d P(x_j \mid y = c)}{P(\mathbf{x})}$$

### 2. Multinomial Model for Text (with Laplace Smoothing)
$$P(w_j \mid y = c) = \frac{N_{cj} + \alpha}{N_c + \alpha |V|}$$
Where $\alpha = 0.1$ prevents zero probability penalties for rare words absent from specific training classes.

### 3. Gaussian Model for Continuous Variables
$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} \right)$$

---

## 🧪 Experiments Overview

### 1. `Spam_Filter_NB.ipynb`
* **Task:** SMS Spam Detection (Spam vs. Ham).
* **Pipeline:** Raw SMS text $\to$ punctuation removal & lowercasing $\to$ word tokenization $\to$ Term Frequency vectorization $\to$ `MultinomialNB(alpha=0.1)`.
* **Diagnostics:**
  * Confusion matrix heatmap showing true positive vs. false positive spam rates.
  * **Top 10 Influential Spam Words:** Horizontal bar chart of log-probability features indicating strongest spam signals (e.g. *free*, *claim*, *call*, *win*).
  * **Word Frequency Comparison:** Frequency counts comparing distribution between Spam and Ham corpora.

### 2. `Iris_Gaussian_NB.ipynb`
* **Task:** Multi-class classification on the canonical Fisher's Iris dataset (Setosa, Versicolor, Virginica).
* **Pipeline:** 4 continuous morphological features $\to$ Standard scaling $\to$ `GaussianNB()`.
* **Diagnostics:**
  * Macro-averaged Precision, Recall, and $F_1$-score.
  * **Probability Density Plots (KDE):** Class-conditional Gaussian distributions $P(\text{Petal Length} \mid \text{Species})$.
  * **2D Decision Boundary:** Meshgrid contour plot visualizing non-linear elliptical decision boundaries separating the three flower species.
  * Direct comparison with Logistic Regression.

---

## 📂 Notebook Implementations
* `Spam_Filter_NB.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/Spam_Filter_NB.ipynb)
* `Iris_Gaussian_NB.ipynb`
  * [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-machine-learning-labs/blob/main/04_naive_bayes_classification/Iris_Gaussian_NB.ipynb)
