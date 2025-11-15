import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# === Load dataset ===
data_path = "processed_heart_failure_data.csv"
data = pd.read_csv(data_path)

print("[INFO] Columns in dataset:", list(data.columns))

# === Kolom target ===
target_col = "HeartDisease"
if target_col not in data.columns:
    raise KeyError(f"[ERROR] Target column '{target_col}' not found in dataset.")

# === Pisahkan fitur dan target ===
X = data.drop(columns=[target_col])
y = data[target_col]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === MLflow setup ===
mlflow.set_experiment("heart_failure_experiment_basic")

with mlflow.start_run() as run:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", acc)

    # === WAJIB: log model dengan artifact_path ===
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:3]
    )

    print(f"[INFO] Accuracy: {acc:.4f}")
    print(f"[INFO] Model logged with run_id = {run.info.run_id}")
