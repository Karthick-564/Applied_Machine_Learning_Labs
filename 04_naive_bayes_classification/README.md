# 04 · Naïve Bayes Classifiers & NLP Fundamentals

> **Core Focus:** Bayesian probability, conditional independence assumptions, text tokenization, vocabulary vectorization, and spam/sentiment text classification.

---

## 📐 Bayes' Theorem & Conditional Independence

$$P(y \mid \mathbf{x}) = \frac{P(y) \prod_{i=1}^d P(x_i \mid y)}{P(\mathbf{x})}$$

### Multinomial Naïve Bayes with Laplace Smoothing
For word tokens $w_i$ in document $d$:
$$P(w_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$$
Where $\alpha = 1.0$ (Laplace smoothing) prevents zero-frequency zero probability estimates for out-of-vocabulary terms.

---

## 📂 Notebook Implementations
* `01_text_preprocessing_and_tokenization.ipynb`: Lowercasing, punctuation stripping, stopword elimination, and Term Frequency feature extraction.
* `02_multinomial_naive_bayes_classifier.ipynb`: Training, evaluating, and plotting the confusion matrix and precision/recall curves for document classification.
