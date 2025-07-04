\# 🏠 Housing Price Prediction using MLOps



This project is part of \*\*MLOps Assignment 1\*\* and focuses on implementing a machine learning pipeline for \*\*house price prediction\*\* using the Boston Housing dataset. The solution involves modular Python code, multiple regression models, and a reproducible workflow using GitHub Actions.



---



\## ✅ Tasks Completed So Far



\- \[x] Set up a Python 3.13 environment using \*\*Conda\*\*

\- \[x] Installed necessary libraries (`pandas`, `numpy`, `scikit-learn`, etc.)

\- \[x] Created a \*\*modular project structure\*\*

\- \[x] Loaded the \*\*Boston Housing dataset\*\* from original source (CMU stats server)

\- \[x] Implemented \*\*three regression models\*\*:

&nbsp; - Linear Regression

&nbsp; - Decision Tree Regressor

&nbsp; - Random Forest Regressor

\- \[x] Evaluated all models using:

&nbsp; - \*\*Mean Squared Error (MSE)\*\*

&nbsp; - \*\*R² Score\*\*

\- \[x] Created Git repository and pushed the code to \*\*GitHub\*\*

\- \[x] Created and switched to `reg` branch (for regression pipeline)



---



\## 🔧 Project Structure



HousingRegression/

│

├── utils.py # Loads and prepares the dataset

├── regression.py # Contains regression training and evaluation

├── requirements.txt # Python dependencies

├── README.md # Project documentation

└── test.py # Script to test data loading





---



\## 📊 Dataset



The Boston Housing dataset is loaded manually as per the instructions using `pandas`. It contains features like:

\- CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, etc.

\- Target: \*\*MEDV\*\* (Median value of owner-occupied homes in $1000s)



---



\## 📌 Next Steps



\- Implement \*\*hyperparameter tuning\*\* in a new branch `hyper`

\- Set up \*\*GitHub Actions (CI)\*\* to automate testing

\- Merge branches back to `main` with documentation

\- Submit report with GitHub link and comparison metrics



---

