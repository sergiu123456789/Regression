# Predict Employee Salary from Years of Experience (sklearn + Amazon SageMaker Linear Learner)

This is an end-to-end workflow: importing libraries, EDA & visualization, train/test split, training & evaluating a **scikit-learn Linear Regression** model, then training, deploying, and testing a **SageMaker Linear Learner** model.

---

## 1) Prerequisites & Setup

- An AWS account with SageMaker Studio/Notebook access.
- An IAM role with SageMaker + S3 permissions.
- A CSV dataset with at least two columns:
  - `YearsExperience` (feature)
  - `Salary` (label/target)

**Assumptions:**
- Column names are `YearsExperience` and `Salary`.

## 2) Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

import sagemaker
import boto3
from sagemaker import Session
```

---

## 3) Load Data

```python
salary_df = pd.read_csv('salary.csv')

print(df.head())
print(df.describe())
print(df.dtypes)
```
## 4) Exploratory Data Analysis (EDA) & Visualization

sns.heatmap(salary_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

---

## 5) Create Train/Test Splits

```python
X = df[["YearsExperience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 6) Train a scikit-learn Linear Regression Model

```python
regresssion_model_sklearn = LinearRegression(fit_intercept = True)
regresssion_model_sklearn.fit(X_train, y_train)
```

## 7) Evaluate the sklearn Model

```python
y_predict = regresssion_model_sklearn.predict(X_test)

```

Visualization:

```python
plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regresssion_model_sklearn.predict(X_train), color = 'red')
plt.ylabel('Salary')
plt.xlabel('Number of Years of Experience')
plt.title('Salary vs. Years of Experience')
```

---

## 8) Prepare Data for SageMaker

```python
sagemaker_session = sagemaker.Session()
bucket = Session().default_bucket()

prefix = 'linear_learner'

role = sagemaker.get_execution_role()

---

## 9) Train SageMaker Linear Learner

```python
container = get_image_uri(boto3.Session().region_name, 'linear-learner')

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)

linear.set_hyperparameters(feature_dim = 1,
                           predictor_type = 'regressor',
                           mini_batch_size = 5,
                           epochs = 50,
                           num_models = 32,
                           loss = 'absolute_loss')

linear.fit({'train': s3_train_data})
```

---

## 10) Deploy Model to Endpoint

```python
linear_regressor = linear.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m4.xlarge')

---

## 11) Invoke Endpoint

Single prediction:

```python
result = linear_regressor.predict(X_test)
```

## 12) Visualize test set results
```python
  plt.scatter(X_test, y_test, color = 'gray')
  plt.plot(X_test, predictions, color = 'red')
  plt.xlabel('Years of Experience (Testing Dataset)')
  plt.ylabel('salary')
  plt.title('Salary vs. Years of Experience')
 ``` 
## 13) Cleanup

```python
linear_regressor.delete_endpoint()
```

---

## 14) Tips

- Label must be the first column in CSV for SageMaker built-in algorithms.
- Remove CSV headers for training.
- Always delete endpoints when finished to avoid costs.
- Consider hyperparameter tuning for best results.

---

## 15) Sample Dataset

`salary_data.csv`:

```csv
YearsExperience,Salary
1,40000
2,45000
3,50000
4,60000
5,65000
6,70000
7,80000
8,90000
9,95000
10,110000
```
