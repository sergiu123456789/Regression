
# 📊 Multiple Linear Regression: Predict the Profit of Startups

## Problem Statement
A venture capital fund wants to predict **which startup to invest in**.  
They have data on several startups, including:
- **R&D Spend**
- **Administration Spend**
- **Marketing Spend**
- **State (Location)**
- **Profit** (Target variable)

The goal is to **build a model that predicts profit for a new startup** based on its spending and state.

---

## Approach

### 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
```

### 2. Importing the Dataset
```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # All columns except the last (features)
y = dataset.iloc[:, -1].values   # Last column (target: Profit)
```

### 3. Encoding Categorical Data (State column)
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
```

### 4. Splitting the Data into Training and Test Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

## Model

We use **Multiple Linear Regression** since multiple factors affect profit.  

**General equation:**

\[
Profit = \beta_0 + \beta_1(R\&D) + \beta_2(Admin) + \beta_3(Marketing) + \beta_4(State) + \epsilon
\]

---

### 5. Training the Multiple Linear Regression Model
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### 6. Predicting the Test Set Results
```python
y_pred = regressor.predict(X_test)

# Compare predictions and actual values
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
```

---

## Prediction Example

Suppose we want to predict profit for:
- **R&D Spend = $120,000**  
- **Administration = $80,000**  
- **Marketing = $50,000**  
- **State = California**

### 7. Making a Single Prediction
```python
# Assuming California is the third dummy variable in OneHotEncoder: [0, 0, 1]
new_data = [[0, 0, 1, 120000, 80000, 50000]]
prediction = regressor.predict(new_data)
print(f"Predicted Profit: ${prediction[0]:,.2f}")
```

---

## Interpreting the Model

### 8. Getting the Final Regression Equation
```python
b0 = regressor.intercept_
coefficients = regressor.coef_

print("Final regression equation:\n")
equation = f"y = {b0:.2f}"
for i, b in enumerate(coefficients):
    equation += f" + ({b:.2f})*x{i+1}"
print(equation)
```

---

## Benefits
- Helps investors make **data-driven decisions**.
- Identifies which factors (R&D, Marketing, State, etc.) have the **highest impact** on profitability.
- Reduces risk by forecasting returns before investing.

---

## Next Steps
1. Collect more startup data for better accuracy.
2. Extend the model to include other factors (team size, industry type, competition).
3. Explore **non-linear models** (Random Forest, XGBoost) for improved predictions.

---

## ✅ Summary
- Built a **Multiple Linear Regression** model.  
- Encoded categorical variables (State).  
- Predicted profits for new startups.  
- Extracted and interpreted the **final regression equation** with coefficients.  
