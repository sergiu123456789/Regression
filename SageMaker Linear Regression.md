Predict Employee Salary from Years of Experience (sklearn + Amazon SageMaker Linear Learner)

1) Prerequisites & Setup
An AWS account with SageMaker Studio/Notebook access.

An IAM role with SageMaker + S3 permissions.

A CSV dataset with at least two columns, e.g.:

YearsExperience (feature)

Salary (label/target)

# Core
import pandas as pd
import numpy as np

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

# Modeling (sklearn)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# SageMaker
import sagemaker
import boto3
from sagemaker import Session

# Quick schema check
print(df.head())
print(df.describe())
print(df.dtypes)

Expected columns

YearsExperience (numeric)

Salary (numeric)

4) Exploratory Data Analysis (EDA) & Visualization

# Scatter plot
plt.figure(figsize=(6,4))
plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()

# Correlation
corr = df[["YearsExperience","Salary"]].corr()
print("Correlation matrix:\n", corr)
Interpretation:

A strong positive correlation suggests a linear model may fit well.

Look for outliers or obvious non-linear patterns.

5) Create Train/Test Splits
python
Copy
Edit
X = df[["YearsExperience"]].values  # 2D for sklearn
y = df["Salary"].values

# 80/20 split with a fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape
6) Train a scikit-learn Linear Regression Model
python
Copy
Edit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("Intercept:", linreg.intercept_)
print("Coefficient:", linreg.coef_)  # per year of experience
7) Evaluate the Trained sklearn Model
python
Copy
Edit
y_pred = linreg.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2:  {r2:.4f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
(Optional) Visualize predictions:

python
Copy
Edit
plt.figure(figsize=(6,4))
plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, y_pred, marker="x", label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Actual vs Predicted (sklearn Linear Regression)")
plt.legend()
plt.show()
8) Prepare Data for SageMaker (S3 upload)
SageMaker’s Linear Learner built-in algorithm accepts CSV where the first column is the label. We’ll create train/validation splits in that format and upload to S3.

python
Copy
Edit
# Create a copy of splits in label-first CSV format
train_csv = pd.DataFrame({"Salary": y_train.flatten(), "YearsExperience": X_train.flatten()})
val_csv   = pd.DataFrame({"Salary": y_test.flatten(),  "YearsExperience": X_test.flatten()})

# Save locally
os.makedirs("sm_data", exist_ok=True)
train_path = "sm_data/train.csv"
val_path   = "sm_data/validation.csv"

train_csv.to_csv(train_path, index=False, header=False)   # no header for built-ins
val_csv.to_csv(val_path, index=False, header=False)

# SageMaker session & role
session = Session()
region = session.boto_region_name
try:
    role = get_execution_role()
except Exception:
    # Fallback if running locally: set your role ARN
    role = "arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>"

# S3 bucket/prefix
default_bucket = session.default_bucket()
prefix = "salary-linear-learner"

# Upload to S3
s3_train = session.upload_data(path=train_path, bucket=default_bucket, key_prefix=f"{prefix}/train")
s3_val   = session.upload_data(path=val_path,   bucket=default_bucket, key_prefix=f"{prefix}/validation")

print("S3 train:", s3_train)
print("S3 val:",   s3_val)
9) Configure and Train SageMaker Linear Learner (Regressor)
python
Copy
Edit
# Get the container image for Linear Learner
image_uri = image_uris.retrieve(framework="linear-learner", region=region)

# Output path for model artifacts
output_path = f"s3://{default_bucket}/{prefix}/output"

# Estimator
ll_estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=output_path,
    sagemaker_session=session,
)

# Hyperparameters: predictor_type='regressor' for regression problems
ll_estimator.set_hyperparameters(
    predictor_type="regressor",   # regression
    epochs=50,                    # adjust as needed
    mini_batch_size=32,           # tune based on dataset size
    loss="squared_loss"           # default for regression
)

# Training inputs (CSV; label must be first column, no header)
train_input = TrainingInput(
    s3_data=s3_train,
    content_type="text/csv"
)
val_input = TrainingInput(
    s3_data=s3_val,
    content_type="text/csv"
)

ll_estimator.fit({"train": train_input, "validation": val_input})
Notes

Increase instance_type or epochs for larger datasets.

Consider tuning with SageMaker Automatic Model Tuning for best hyperparameters.

10) Deploy the Trained Linear Learner Model to a Real-Time Endpoint
python
Copy
Edit
# Deploy to a managed endpoint
predictor = ll_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=CSVSerializer(),          # send CSV rows (no header)
    deserializer=JSONDeserializer()      # receive JSON predictions
)
endpoint_name = predictor.endpoint_name
print("Endpoint:", endpoint_name)
11) Invoke the Endpoint (Prediction)
Send one or multiple examples as a CSV string (no header). Our model expects the same feature order as training (just YearsExperience).

python
Copy
Edit
# Single record prediction (e.g., 5.0 years of experience)
test_experience = [[5.0]]
payload_single = "\n".join([",".join(map(str, row)) for row in test_experience])

response_single = predictor.predict(payload_single)
print("Prediction (single):", response_single)
Batch a few values:

python
Copy
Edit
# Multiple records
X_new = [[0.5], [2.0], [5.0], [10.0]]
payload_multi = "\n".join([",".join(map(str, row)) for row in X_new])

response_multi = predictor.predict(payload_multi)
print("Prediction (batch):", response_multi)
Interpreting the response
For Linear Learner regressors, the response typically contains a "predictions" list with "score" values. Example shape:

json
Copy
Edit
{"predictions": [{"score": 40123.1}, {"score": 51234.7}, ...]}
12) Compare SageMaker vs sklearn Predictions (Optional)
python
Copy
Edit
# Compare on the same X_test
# 1) sklearn predictions (already computed as y_pred)
# 2) SageMaker predictions
payload_test = "\n".join([",".join(map(str, row)) for row in X_test])
pred_ll = predictor.predict(payload_test)

sm_scores = np.array([p["score"] for p in pred_ll["predictions"]])

r2_sm  = r2_score(y_test, sm_scores)
mse_sm = mean_squared_error(y_test, sm_scores)
rmse_sm = np.sqrt(mse_sm)
mae_sm = mean_absolute_error(y_test, sm_scores)

print(f"SageMaker Linear Learner — R^2: {r2_sm:.4f}, RMSE: {rmse_sm:.2f}, MAE: {mae_sm:.2f}")
print(f"sklearn Linear Regression — R^2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
13) (Important) Endpoint Cleanup
Real-time endpoints incur cost while running. Delete when done:

python
Copy
Edit
# Delete endpoint + config to stop charges
session.delete_endpoint(predictor.endpoint_name)
session.delete_endpoint_config(predictor.endpoint_name)
14) Tips & Troubleshooting
Data format: For Linear Learner with CSV, ensure label is first and no header.

Scaling: Linear models generally don’t require scaling for a single feature, but if you expand features, consider normalization.

Outliers: Large outliers in salary can skew linear fits—consider robust regression or log-transforming the target if appropriate.

Model choice: If the relationship is not linear, try polynomial features (sklearn PolynomialFeatures) or tree-based models (XGBoost on SageMaker).

Permissions: If get_execution_role() fails locally, provide a valid IAM role ARN with SageMaker & S3 permissions.

Costs: Prefer smaller instance types (ml.t2.medium/ml.m5.large) for demos; scale up for production.

15) Minimal CSV Example (for quick testing)
Save as data/salary_data.csv:

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
Then re-run the steps above.

