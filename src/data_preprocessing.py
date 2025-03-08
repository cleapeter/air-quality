import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(config):
    df = pd.read_csv(config["data"]["clean_path"])
    return df


def preprocess_data(df, target_column):
    scaler = StandardScaler()

    # Ignore target and time columns
    cols_to_scale = [col for col in df.columns if col not in [target_column, "Timestamp", "Hour", "DayOfWeek", "Month"]]

    # Standardize numerical columns
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Create dummies for time columns
    df = pd.get_dummies(df, columns=["Hour", "DayOfWeek", "Month"])

    os.makedirs("./models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return df.drop(columns=[target_column]), df[target_column]


def split_data(X, y, target_name, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define target-specific subfolder
    output_dir = f"data/processed/split/{target_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save split data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    df_X, df_y = preprocess_data(df, "CO_reference_mg_per_m3")
    print(df_X.head())

    X_train, X_test, y_train, y_test = split_data(df_X, df_y, "CO_reference_mg_per_m3")
    print(X_train.head())
