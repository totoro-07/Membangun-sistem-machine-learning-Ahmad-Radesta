import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# === Load dataset ===
data_path = "processed_heart_failure_data.csv"
data = pd.read_csv(data_path)

print("[INFO] Columns in dataset:", list(data.columns))

# === Kolom target dataset hasil preprocessing ===
target_col = "HeartDisease"

if target_col not in data.columns:
    raise KeyError(f"[ERROR] Target column '{target_col}' not found in dataset.")

X = data.drop(target_col, axis=1)
y = data[target_col]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === MLflow settings (Basic) ===
mlflow.set_experiment("heart_failure_experiment_basic")
mlflow.sklearn.autolog()  # autolog aktif

# Tidak ada 'with mlflow.start_run()' di sini
model = RandomForestClassifier()
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("[INFO] Accuracy:", acc)

# Log hasil ke MLflow
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")
