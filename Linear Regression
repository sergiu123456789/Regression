# 🎓 Linear Regression prediction of salary of an employee

**Linear Regression** is one of the most commonly used statistical modeling methods in data science.

### ✅ 1. Importing Libraries and Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Years of Experience
y = dataset.iloc[:, -1].values   # Salary
```

### 2. Splitting the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
```

### 3. Training the Simple Linear Regression Model

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### 4. Predicting the Test Set Results

```python
y_pred = regressor.predict(X_test)
```

### 5. Visualizing the Training Set Results

```python
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### 6. Visualizing the Test Set Results

```python
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### 7. Making a Single Prediction

```python
# Predict the salary for an employee with 12 years of experience
print(regressor.predict([[12]]))
```

### 8. Getting the Final Regression Equation

```python
b0 = regressor.intercept_
b1 = regressor.coef_[0]

print(f"The final regression equation is: y = {b0:.2f} + {b1:.2f} * x")
```

---

## ✅ Summary

- Train a simple linear regression model.
- Visualize both training and test set results.
- Make a prediction for a specific input.
- Extract and interpret the final regression equation.
