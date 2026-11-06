# 🏡 House Price Prediction: Regression Models Comparison

A comparative study of core regression algorithms used to predict residential property prices based on key features (square footage, distance to city center, house age, neighborhood rating, etc.).

---

## 📌 Project Overview

The main objective of this project is to build, evaluate, and benchmark multiple machine learning models to accurately estimate housing values.

- **Target Variable:** `price`
- **Dataset Size:** 500 samples, 8 numerical features
- **Methodology:** 80/20 Train-Test split combined with 5-Fold Cross-Validation, evaluated across $R^2$, MAE, MSE, and RMSE metrics.

---

## 📊 Results & Model Benchmarks

| Model | MAE ($) | RMSE ($) | Test R² | CV R² (Mean) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 19,225.85 | 25,268.05 | **0.8938** | **0.8685** |
| **Lasso Regression** | 19,225.88 | 25,268.04 | 0.8938 | 0.8685 |
| **Ridge Regression** | 19,248.44 | 25,287.05 | 0.8936 | 0.8686 |
| **Elastic Net** | 19,739.39 | 25,834.72 | 0.8890 | 0.8670 |
| **Polynomial Regression (Degree 2)** | 21,450.13 | 27,163.31 | 0.8772 | 0.8516 |

---

## 🛠️ Machine Learning Pipeline

1. **EDA & Visualization:** Correlation heatmaps and target-to-feature relationship plots.
2. **Preprocessing:** Feature scaling via `StandardScaler` integrated directly into Scikit-Learn `Pipeline` objects to prevent data leakage.
3. **Cross-Validation:** `KFold(n_splits=5, shuffle=True)` applied strictly to training sets.
4. **Evaluation:** Cumulative benchmarking across standard error metrics and coefficient of determination ($R^2$).

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR-USERNAME/housing-price-prediction.git](https://github.com/YOUR-USERNAME/housing-price-prediction.git)
cd housing-price-prediction
