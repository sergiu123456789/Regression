# 📊 Multiple Linear Regression: Predict the profit of certain state companies based on their spendings

## 🧩 Problem Statement

The **HR department** is evaluating a job applicant who claims to have earned a certain **salary** at their previous company. To determine if this expectation is **reasonable**, HR wants to use a data-driven method to predict expected salary based on:

- **Position Level**
- **Years of Experience**

### 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

### 2. Importing the Dataset

```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # All columns except the last (features)
y = dataset.iloc[:, -1].values   # Last column (target: Profit)

### 3. Encoding Categorical Data (State column)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

### 4. Splitting the Data into Training and Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### 5. Training the Multiple Linear Regression Model

regressor = LinearRegression()
regressor.fit(X_train, y_train)

### 6. Predicting the Test Set Results

y_pred = regressor.predict(X_test)

# Compare predictions and actual values
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

### 7. Making a Single Prediction

```python
# Input: State = California, R&D = 160000, Admin = 130000, Marketing = 300000
# Assuming California is the third dummy variable (position 2) in OneHotEncoder
# OneHotEncoded: [0, 0, 1], followed by numerical values
new_data = [[0, 0, 1, 160000, 130000, 300000]]
prediction = regressor.predict(new_data)
print(f"Predicted Profit: ${prediction[0]:,.2f}")

### 8. Getting the Final Regression Equation

```python
b0 = regressor.intercept_
coefficients = regressor.coef_

print(f"Final regression equation:\n")
equation = f"y = {b0:.2f}"
for i, b in enumerate(coefficients):
    equation += f" + ({b:.2f})*x{i+1}"
print(equation)

## ✅ Summary

- Train a multiple linear regression model
- Encode categorical variables
- Predict values from unseen inputs
- Extract and interpret the final regression equation with coefficients
