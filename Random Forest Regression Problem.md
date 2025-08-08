# 🌲 Random Forest Regression Problem: Validating Salary Expectations

## 🧩 Problem Statement

The **HR department** is evaluating a job applicant who claims to have earned a certain **salary** at their previous company. To determine if this expectation is **reasonable**, HR wants to use a data-driven method to predict expected salary based on:

- **Position Level**
- **Years of Experience**

Using **Random Forest Regression**, HR can compare the candidate's claimed salary with industry trends for similar roles and experience levels.

---

## 🎯 Objective

> Use **Random Forest Regression** to model the relationship between:
> - Position Level  
> - Experience  
>
> and the **Salary**, in order to predict the likely salary range and verify the realism of a salary claim.

---

## 🌲 Why Random Forest Regression?

- Handles **non-linear relationships** extremely well.
- Robust to **outliers and overfitting**.
- Works great with both small and large datasets.
- Provides **feature importance** to show which variables impact salary the most.

---

## 🧪 Dataset Example

| Position Level | Experience (Years) | Salary (USD) |
|----------------|--------------------|--------------|
| 1              | 1                  | 45,000       |
| 3              | 3                  | 60,000       |
| 5              | 5                  | 85,000       |
| 7              | 7                  | 120,000      |
| 10             | 10                 | 200,000      |

We'll train a Random Forest model using this data to learn the salary trends.

---

## 🤖 Candidate's Claim

Suppose the candidate claims:
- **Position Level**: 6  
- **Experience**: 6 years  
- **Claimed Salary**: $150,000

We use the trained Random Forest Regressor to estimate the expected salary:

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Train model
model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X, y)

# Predict for candidate
predicted_salary = model.predict([[6, 6]])
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
```

If the predicted salary is **significantly lower or higher** than the claimed **$150,000**, HR might want to investigate further or adjust their offer accordingly.

---

## ✅ Outcome

- Objective evaluation of candidate salary claims
- Data-informed decisions in hiring and compensation
- Better alignment with real-world salary trends

---

## 🛠️ Advantages of Using Random Forest

- No need for feature scaling
- Can model complex relationships
- Offers **feature importance** insight:
  ```python
  import matplotlib.pyplot as plt

  importance = model.feature_importances_
  plt.bar(['Position Level', 'Experience'], importance)
  plt.title("Feature Importance")
  plt.show()
  ```

---

## 🧠 Summary

Using **Random Forest Regression**, HR can:
- Reliably predict salaries based on past data
- Handle non-linear patterns without heavy preprocessing
- Support fair, consistent, and transparent hiring practices

📌 This approach adds **trustworthy machine learning insights** to the HR process, reducing bias and improving salary offer accuracy.
