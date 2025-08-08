# 🌳 Decision Tree Regression Problem: Validating Salary Expectations

## 🧩 Problem Statement

The **HR department** is reviewing a candidate who claims to have earned a certain **salary** at a previous job. To verify if this salary expectation is **reasonable**, HR wants to analyze:

- The **position level** of the candidate
- Their **years of experience**
- The **salary they claimed**

Using **Decision Tree Regression**, HR can model past salary data and determine if the candidate's claimed salary is consistent with typical compensation for similar profiles.

---

## 🎯 Objective

> Use **Decision Tree Regression** to predict salaries based on:
> - Position Level  
> - Experience  
>
> and determine if a new candidate's salary claim falls within expected values.

---

## 🌳 Why Decision Tree Regression?

- Captures **non-linear relationships** without requiring feature scaling
- Easy to interpret and visualize
- Automatically splits data based on important **thresholds**
- Handles outliers and skewed data well

---

## 🧪 Example Dataset

| Position Level | Experience (Years) | Salary (USD) |
|----------------|--------------------|--------------|
| 1              | 1                  | 45,000       |
| 3              | 3                  | 60,000       |
| 5              | 5                  | 85,000       |
| 7              | 7                  | 120,000      |
| 10             | 10                 | 200,000      |

We train a **Decision Tree Regressor** on this dataset to learn salary patterns based on role and experience.

---

## 👤 Candidate's Info

Let’s say the candidate reports:
- **Position Level**: 6  
- **Experience**: 6 years  
- **Claimed Salary**: $150,000

We use the trained decision tree model to predict the expected salary:

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Assume X = [[position_level, experience]], y = salary
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict salary for new candidate
new_input = np.array([[6, 6]])
predicted_salary = regressor.predict(new_input)

print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
```

If the predicted salary differs significantly from **$150,000**, it may be flagged for further review.

---

## ✅ Outcome

- Identify **unusual salary expectations**
- Use data-driven predictions to support **fair compensation decisions**
- Enhance the HR decision-making process with interpretable models

---

## 🔍 Additional Tips

- Decision Trees can **overfit** easily — use techniques like:
  - Pruning
  - Setting `max_depth`, `min_samples_split`, etc.
- For smoother predictions, consider **Random Forest Regression**

---

## 🧠 Summary

**Decision Tree Regression** offers an interpretable and effective way for HR teams to:

- Model salary expectations
- Predict fair compensation
- Detect discrepancies in candidate-reported salaries

This helps make the hiring process more transparent and data-driven.

