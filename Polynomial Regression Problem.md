# 📈 Polynomial Regression Problem: Validating Salary Expectations

## 🧩 Problem Statement

The **HR department** is evaluating a job candidate who claims to have earned a specific **salary** at their previous company.

To verify whether the candidate's **salary expectation is realistic**, HR wants to analyze:

- The **position level** (e.g., 1 for Junior, 5 for Mid-Level, 10 for Senior)
- The **years of experience**
- The **actual salary claimed**

Using this information, HR aims to **predict the expected salary** for someone in a similar role using a trained **Polynomial Regression model**. If the predicted salary is significantly different from what the candidate claimed, it may raise a red flag.

---

## 🎯 Objective

> Use **Polynomial Regression** to model the non-linear relationship between a candidate’s:
> - Position Level
> - Experience
> 
> and their **salary**, and then determine if their salary claim is consistent with typical industry patterns.

---

## 📌 Why Polynomial Regression?

Unlike **Linear Regression**, which assumes a straight-line relationship, **Polynomial Regression** can capture **curved trends** — which are common in salary structures where increases are not always linear with experience or position level.

---

## 🧪 Dataset Example

| Position Level | Experience (Years) | Salary (USD) |
|----------------|--------------------|--------------|
| 1              | 1                  | 45,000       |
| 3              | 3                  | 60,000       |
| 5              | 5                  | 85,000       |
| 7              | 7                  | 120,000      |
| 10             | 10                 | 200,000      |

We train a Polynomial Regression model on such historical data.

---

## 🤖 Model Usage Example

Let’s say the candidate claims:
- Position Level: **6**
- Experience: **6 years**
- Claimed Salary: **$150,000**

We feed these inputs into our trained Polynomial Regression model:

```python
predicted_salary = model.predict([[6, 6]])
```

If `predicted_salary` is significantly **lower** or **higher** than `$150,000`, HR may want to further **investigate the claim**.

---

## ✅ Outcome

- 🔍 Detect salary claims that **deviate from industry norms**
- 📊 Use historical data to **support compensation decisions**
- 🧠 Add a data-driven layer to the HR hiring process

---

## 📂 Notes

- Model performance can be improved using **cross-validation**, **feature scaling**, and **higher-degree polynomials**
- Additional features such as **company size**, **location**, or **industry** can be included for better predictions

