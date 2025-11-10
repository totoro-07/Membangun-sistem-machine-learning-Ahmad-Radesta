import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# AUTOMATED PREPROCESSING PIPELINE
# ---------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Dataset not found at path: {path}")


def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Remove outliers using IQR method.
    Only applied to numeric columns that truly exist.
    """
    df_clean = df.copy()

    for col in numeric_cols:
        if col not in df_clean.columns:
            print(f"[WARNING] Column '{col}' not found. Skipping outlier removal.")
            continue

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        after = df_clean.shape[0]

        print(f"[INFO] Outlier removal for {col}: {before - after} rows removed.")

    return df_clean


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to categorical features.
    Drop columns that do not exist to avoid errors.
    """
    cols_found = [c for c in categorical_cols if c in df.columns]

    if len(cols_found) < len(categorical_cols):
        missing = set(categorical_cols) - set(cols_found)
        print(f"[WARNING] Missing categorical columns skipped: {missing}")

    df_encoded = pd.get_dummies(df, columns=cols_found, drop_first=True)
    print("[INFO] One-Hot Encoding applied.")

    return df_encoded


def scale_features(X: pd.DataFrame):
    """
    Standardize numerical features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[INFO] Feature scaling completed.")
    return X_scaled, scaler


def preprocess_pipeline(path: str):
    """
    Full preprocessing pipeline:
    1. Load dataset
    2. Remove outliers
    3. Encode categorical
    4. Train-test split
    5. Scale features
    6. Save processed dataset
    """

    df = load_dataset(path)

    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    df = remove_outliers_iqr(df, numeric_cols)
    df = encode_categorical(df, categorical_cols)

    # safety check
    if "HeartDisease" not in df.columns:
        raise KeyError("[ERROR] 'HeartDisease' column not found in dataset.")

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"[INFO] Train-test split completed: {X_train.shape[0]} train samples")

    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = scaler.transform(X_test)

    processed_df = pd.DataFrame(X_train_scaled)
    processed_df["HeartDisease"] = y_train.values

    output_path = "processed_heart_failure_data.csv"
    processed_df.to_csv(output_path, index=False)

    print(f"[INFO] Processed dataset saved at: {output_path}")
    print("[INFO] Preprocessing pipeline finished successfully.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------
# Example execution (comment / uncomment)
# ---------------------------------------------
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline("../Heart Failure Prediction Dataset.csv")
print("Preprocessing completed.")
