# 🏠 Housing Price Prediction (MLOps Assignment 1)

This repository contains a complete MLOps pipeline to predict **Boston housing prices** using classical regression models. It includes **modular Python code**, **GitHub Actions CI**, and **hyperparameter tuning** using `GridSearchCV`.

---

## 🔗 GitHub Repository Link (for Report)
[https://github.com/Vikranth-Venkateswar/HousingRegression](https://github.com/Vikranth-Venkateswar/HousingRegression)

---

## ✅ Project Progress Summary

- [x] Set up `conda` environment (`mlops_assignment`)
- [x] Loaded Boston Housing dataset manually (as required)
- [x] Implemented 3 regression models:
  - Linear/Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- [x] Compared models using **MSE** and **R²**
- [x] Created separate `reg` and `hyper` branches
- [x] Tuned 3+ hyperparameters per model with `GridSearchCV`
- [x] Created and configured **CI/CD** with GitHub Actions
- [x] Merged `reg` and `hyper` to `main`
- [x] Prepared final report for submission

---

## 📁 Folder Structure

HousingRegression/
│
├── .github/
│ └── workflows/
│ └── ci.yml # GitHub Actions workflow for CI
│
├── utils.py # Loads and prepares the dataset
├── regression.py # Trains and evaluates baseline models
├── hyperparameter_tuning.py # Performs hyperparameter tuning
├── requirements.txt # List of required Python packages
├── README.md 


---

## 📊 Dataset Used

- **Boston Housing Dataset**  
- Loaded manually using code from:
  http://lib.stat.cmu.edu/datasets/boston  
- Target variable: `MEDV` (Median house value in $1000s)

---

## 📈 Evaluation Metrics

All models are evaluated using:

- **Mean Squared Error (MSE)**
- **R² Score**

A comparison is included in the final report PDF.

---

## 🔧 How to Run

```bash
# Activate conda environment
conda activate mlops_assignment

# Install dependencies
pip install -r requirements.txt

# Run baseline regression
python regression.py

# Run hyperparameter tuning
python hyperparameter_tuning.py



CI/CD Setup
GitHub Actions CI is triggered on every push to:

reg branch

hyper branch

It installs dependencies and runs model code automatically.
