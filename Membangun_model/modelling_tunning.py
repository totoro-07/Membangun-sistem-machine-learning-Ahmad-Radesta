import dagshub
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Inisialisasi koneksi ke DagsHub ===
dagshub.init(repo_owner='totoro-07',
             repo_name='my-first-repo',
             mlflow=True)

# === Load dataset ===
data_path = "processed_heart_failure_data.csv"
data = pd.read_csv(data_path)
print("[INFO] Columns in dataset:", list(data.columns))

# === Kolom target ===
target_col = "HeartDisease"
if target_col not in data.columns:
    raise KeyError(f"[ERROR] Target column '{target_col}' not found in dataset.")

X = data.drop(target_col, axis=1)
y = data[target_col]

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Hyperparameter tuning dengan GridSearch ===
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=1
)

with mlflow.start_run(run_name="RandomForest_GridSearch"):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log param & metric
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)

    # Simpan model secara manual
    model_path = "best_model.pkl"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)  # upload ke DagsHub

    print("[INFO] Best Parameters:", grid_search.best_params_)
    print("[INFO] Accuracy:", acc)
    print("[INFO] Model saved and logged as artifact.")
