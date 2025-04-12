import logging
import os

import joblib
import pandas as pd
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


def split_data(data, target_column, train_size=0.7, val_size=0.15):
    try:
        logger.info(f"Splitting data into training, validation, and test sets with target column {target_column}...")
        total_size = len(data)
        train_end = int(total_size * train_size)
        val_end = train_end + int(total_size * val_size)

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        X_val = val_data.drop(columns=[target_column])
        y_val = val_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        logger.info("Data split successfully.")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise


def standardize_data(X_train, X_val, X_test):
    try:
        logger.info("Standardizing the data...")
        scaler = StandardScaler()

        columns_to_standardize = [col for col in X_train.columns if col not in ["Hour", "DayOfWeek", "Month"]]
        X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
        X_val[columns_to_standardize] = scaler.transform(X_val[columns_to_standardize])
        X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])

        # Save the scaler
        os.makedirs("./models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")

        logger.info("Data standardized successfully.")
        return X_train, X_val, X_test
    except Exception as e:
        logger.error(f"Error during data standardization: {e}")
        raise


def create_time_dummies(X_train, X_val, X_test):
    try:
        logger.info("Creating dummy variables for time-related columns...")
        time_columns = ["Hour", "DayOfWeek", "Month"]
        X_train = pd.get_dummies(X_train, columns=time_columns)
        X_val = pd.get_dummies(X_val, columns=time_columns)
        X_test = pd.get_dummies(X_test, columns=time_columns)

        logger.info("Dummy variables created successfully.")
        return X_train, X_val, X_test
    except Exception as e:
        logger.error(f"Error during creation of dummy variables: {e}")
        raise


def preprocess_data(X_train, X_val, X_test):
    try:
        logger.info("Preprocessing data...")
        X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)
        X_train, X_val, X_test = create_time_dummies(X_train, X_val, X_test)
        logger.info("Data preprocessed successfully.")
        return X_train, X_val, X_test
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise
