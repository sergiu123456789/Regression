# 📊 Support Vector Regression (SVR) Problem: Validating Salary Expectations

## 🧩 Problem Statement

The **HR department** is reviewing a job candidate who claims to have earned a certain **salary** at their previous company. To assess whether this claim is **realistic**, HR wants to analyze:

- The **position level**
- The **years of experience**
- The **claimed salary**

By using **Support Vector Regression (SVR)** on historical salary data, HR can determine if this claimed salary aligns with patterns from similar roles and experience levels.

---

## 🎯 Objective

> Use **Support Vector Regression** to model the relationship between:
> - Position Level  
> - Experience  
>
> and **Salary**, in order to assess if the candidate's salary expectation is within a reasonable range.

---

## ❓ Why Support Vector Regression?

- **SVR** is robust for modeling **non-linear relationships**.
- It creates a **margin of tolerance** (epsilon) around the predicted function, making it effective for **real-world noisy data**.
- Works well with **small to medium-sized datasets** where linear models may underperform.

---

## 🧪 Dataset Example

| Position Level | Experience (Years) | Salary (USD) |
|----------------|--------------------|--------------|
| 1              | 1                  | 45,000       |
| 3              | 3                  | 60,000       |
| 5              | 5                  | 85,000       |
| 7              | 7                  | 120,000      |
| 10             | 10                 | 200,000      |

We train an SVR model using this data to learn the salary trends.

---

## 🤖 Candidate's Claim

Let’s say the candidate provides:
- **Position Level**: 6  
- **Experience**: 6 years  
- **Claimed Salary**: $150,000

We use the trained SVR model to predict the expected salary:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Assume X = [[position_level, experience]], y = salary
sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

svr = SVR(kernel='rbf')
svr.fit(X_scaled, y_scaled.ravel())

# Make a prediction
new_input = sc_X.transform([[6, 6]])
predicted_scaled_salary = svr.predict(new_input)
predicted_salary = sc_y.inverse_transform(predicted_scaled_salary.reshape(-1, 1))

print(f"Predicted Salary: ${predicted_salary[0][0]:,.2f}")
```

If the predicted salary is significantly lower or higher than **$150,000**, HR may consider **questioning or validating** the claim further.

---

## ✅ Outcome

- Detect inconsistencies in claimed salaries
- Support compensation decisions with **data-driven models**
- Use **SVR’s non-linear capabilities** to better fit real-world salary distributions

---

## 🔧 Notes

- SVR requires **feature scaling** for proper performance.
- You can tune hyperparameters like:
  - `C` (penalty parameter)
  - `epsilon` (margin of tolerance)
  - `kernel` (linear, rbf, poly)
- Visualizing results with inverse scaling helps communicate findings to HR teams

---

## 🧠 Summary

Using **Support Vector Regression**, HR can:
- Model complex salary trends
- Predict expected salaries
- Validate candidate claims objectively

📌 This adds an intelligent, machine learning-based layer to the hiring process, reducing reliance on assumptions or gut instinct.

