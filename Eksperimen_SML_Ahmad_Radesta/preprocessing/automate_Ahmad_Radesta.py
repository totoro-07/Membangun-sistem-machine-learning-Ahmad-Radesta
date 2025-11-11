import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# AUTOMATED PREPROCESSING PIPELINE
# ---------------------------------------------

def get_dataset_path() -> str:
    """
    Dataset disimpan di folder yang sama dengan folder utama Eksperimen_SML_Ahmad_Radesta.
    Lokasi file: /Eksperimen_SML_Ahmad_Radesta/Heart Failure Prediction Dataset.csv
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # folder Eksperimen_SML_Ahmad_Radesta
    dataset_filename = "Heart Failure Prediction Dataset.csv"
    dataset_path = os.path.join(base_dir, dataset_filename)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"[ERROR] Dataset not found at: {dataset_path}")

    return dataset_path


def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Dataset not found at path: {path}")


def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df_clean = df.copy()

    for col in numeric_cols:
        if col not in df_clean.columns:
            print(f"[WARNING] Column '{col}' not found. Skipping.")
            continue

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        after = df_clean.shape[0]

        print(f"[INFO] Outliers removed from {col}: {before - after}")

    return df_clean


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    present = [c for c in categorical_cols if c in df.columns]
    if len(present) < len(categorical_cols):
        missing = set(categorical_cols) - set(present)
        print(f"[WARNING] Missing categorical columns skipped: {missing}")

    df = pd.get_dummies(df, columns=present, drop_first=True)
    print("[INFO] One-Hot Encoding applied.")
    return df


def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[INFO] Feature scaling completed.")
    return X_scaled, scaler


def preprocess_pipeline(path: str):
    df = load_dataset(path)

    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    df = remove_outliers_iqr(df, numeric_cols)
    df = encode_categorical(df, categorical_cols)

    if "HeartDisease" not in df.columns:
        raise KeyError("[ERROR] 'HeartDisease' not in dataset.")

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"[INFO] Train-test split completed: {X_train.shape[0]} training samples.")

    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed file inside the same preprocessing directory
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "processed_heart_failure_data.csv")

    pd.DataFrame(X_train_scaled).assign(HeartDisease=y_train.values).to_csv(
        output_path, index=False
    )

    print(f"[INFO] Processed dataset saved to: {output_path}")
    print("[INFO] Preprocessing finished successfully.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------
# EXECUTION
# ---------------------------------------------
if __name__ == "__main__":
    dataset_path = get_dataset_path()
    preprocess_pipeline(dataset_path)
    print("[INFO] Preprocessing completed.")
