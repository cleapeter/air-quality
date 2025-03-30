import logging
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_data(config):
    try:
        logger.info(f"Loading data from {config['data']['clean_path']}...")
        df = pd.read_csv(config["data"]["clean_path"])
        df.set_index("Timestamp", inplace=True)
        logger.info("Data loaded successfully.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df, target_column):
    try:
        logger.info("Preprocessing data, scaling and creating dummies...")
        scaler = StandardScaler()

        # Ignore target and time columns
        cols_to_scale = [col for col in df.columns if col not in [target_column, "Hour", "DayOfWeek", "Month"]]

        # Standardize numerical columns
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Create dummies for time columns
        df = pd.get_dummies(df, columns=["Hour", "DayOfWeek", "Month"])

        os.makedirs("./models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")

        logger.info("Data preprocessing completed successfully.")
        return df.drop(columns=[target_column]), df[target_column]
    except KeyError as e:
        logger.error(f"Key error during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise


def split_data(X, y, target_name, test_size=0.2, random_state=42):
    try:
        logger.info(f"Splitting data into training and testing sets for target: {target_name}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Define target-specific subfolder
        output_dir = f"data/processed/split/{target_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Save split data
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        logger.info("Data split successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise
