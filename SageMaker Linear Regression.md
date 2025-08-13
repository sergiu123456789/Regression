# Predict Employee Salary from Years of Experience (sklearn + Amazon SageMaker Linear Learner)

This walkthrough shows an end-to-end workflow: importing libraries, EDA & visualization, train/test split, training & evaluating a **scikit-learn Linear Regression** model, then training, deploying, and testing a **SageMaker Linear Learner** model.

---

## 1) Prerequisites & Setup

- An AWS account with SageMaker Studio/Notebook access.
- An IAM role with SageMaker + S3 permissions.
- A CSV dataset with at least two columns:
  - `YearsExperience` (feature)
  - `Salary` (label/target)

**Assumptions:**
- Local CSV path is `data/salary_data.csv`.
- Column names are `YearsExperience` and `Salary`.

```bash
# (Optional) Create a project folder
mkdir -p data
# Place your dataset at: data/salary_data.csv
2) Import Libraries
python
Copy
Edit
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sagemaker
from sagemaker import Session, get_execution_role, image_uris
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
3) Load Data
python
Copy
Edit
DATA_LOCAL = "data/salary_data.csv"
df = pd.read_csv(DATA_LOCAL)

print(df.head())
print(df.describe())
print(df.dtypes)
If needed:

python
Copy
Edit
df = df.rename(columns={"years_of_exp":"YearsExperience", "pay":"Salary"})
4) Exploratory Data Analysis (EDA) & Visualization
python
Copy
Edit
assert df["YearsExperience"].notnull().all()
assert df["Salary"].notnull().all()

plt.figure(figsize=(6,4))
plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()

print(df[["YearsExperience","Salary"]].corr())
5) Create Train/Test Splits
python
Copy
Edit
X = df[["YearsExperience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
6) Train a scikit-learn Linear Regression Model
python
Copy
Edit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("Intercept:", linreg.intercept_)
print("Coefficient:", linreg.coef_)
7) Evaluate the sklearn Model
python
Copy
Edit
y_pred = linreg.predict(X_test)

print("R^2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
Visualization:

python
Copy
Edit
plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, y_pred, marker="x", label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
8) Prepare Data for SageMaker
python
Copy
Edit
train_csv = pd.DataFrame({"Salary": y_train.flatten(), "YearsExperience": X_train.flatten()})
val_csv   = pd.DataFrame({"Salary": y_test.flatten(), "YearsExperience": X_test.flatten()})

os.makedirs("sm_data", exist_ok=True)
train_csv.to_csv("sm_data/train.csv", index=False, header=False)
val_csv.to_csv("sm_data/validation.csv", index=False, header=False)

session = Session()
region = session.boto_region_name
role = get_execution_role()

default_bucket = session.default_bucket()
prefix = "salary-linear-learner"

s3_train = session.upload_data(path="sm_data/train.csv", bucket=default_bucket, key_prefix=f"{prefix}/train")
s3_val   = session.upload_data(path="sm_data/validation.csv", bucket=default_bucket, key_prefix=f"{prefix}/validation")
9) Train SageMaker Linear Learner
python
Copy
Edit
image_uri = image_uris.retrieve(framework="linear-learner", region=region)
output_path = f"s3://{default_bucket}/{prefix}/output"

ll_estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=output_path,
    sagemaker_session=session,
)

ll_estimator.set_hyperparameters(
    predictor_type="regressor",
    epochs=50,
    mini_batch_size=32,
    loss="squared_loss"
)

train_input = TrainingInput(s3_data=s3_train, content_type="text/csv")
val_input   = TrainingInput(s3_data=s3_val, content_type="text/csv")

ll_estimator.fit({"train": train_input, "validation": val_input})
10) Deploy Model to Endpoint
python
Copy
Edit
predictor = ll_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

print("Endpoint:", predictor.endpoint_name)
11) Invoke Endpoint
Single prediction:

python
Copy
Edit
test_experience = [[5.0]]
payload_single = "\n".join([",".join(map(str, row)) for row in test_experience])
response_single = predictor.predict(payload_single)
print(response_single)
Batch prediction:

python
Copy
Edit
X_new = [[0.5], [2.0], [5.0], [10.0]]
payload_multi = "\n".join([",".join(map(str, row)) for row in X_new])
response_multi = predictor.predict(payload_multi)
print(response_multi)
12) Compare sklearn vs SageMaker
python
Copy
Edit
payload_test = "\n".join([",".join(map(str, row)) for row in X_test])
pred_ll = predictor.predict(payload_test)
sm_scores = np.array([p["score"] for p in pred_ll["predictions"]])

print("SageMaker R^2:", r2_score(y_test, sm_scores))
print("sklearn R^2:", r2_score(y_test, y_pred))
13) Cleanup
python
Copy
Edit
session.delete_endpoint(predictor.endpoint_name)
session.delete_endpoint_config(predictor.endpoint_name)
14) Tips
Label must be the first column in CSV for SageMaker built-in algorithms.

Remove CSV headers for training.

Always delete endpoints when finished to avoid costs.

Consider hyperparameter tuning for best results.

15) Sample Dataset
data/salary_data.csv:

csv
Copy
Edit
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
pgsql
Copy
Edit
