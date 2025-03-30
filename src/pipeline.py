import logging
import os
from datetime import datetime

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from config import load_config
from data_preprocessing import load_data, preprocess_data, split_data
from evaluate import evaluate_model
from train import train_model

# Setup logger
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"{start_time}_pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Air Quality: Random Forest Regressor")


def run_pipeline():
    try:
        logger.info("Pipeline started.")

        logger.info("Loading configuration...")
        config, target_column = load_config()

        model_name = "random_forest_regressor"

        with mlflow.start_run():
            mlflow.log_param("target_column", target_column)

            # Load data
            print("Loading data")
            df = load_data(config)

            # Process data
            print("Processing data")
            df_scaled, df_target = preprocess_data(df, target_column)
            X_train, X_test, y_train, y_test = split_data(df_scaled, df_target, target_column)

            # Train and evaluate model
            print("Training model")
            model_params = config["models"][model_name]
            model = train_model(X_train, y_train, model_name, model_params)
            print("Evaluating model")
            metrics = evaluate_model(X_test, y_test, model)
            mlflow.log_params(model_params)
            mlflow.log_metrics(metrics)

            input_example = pd.DataFrame(X_train.iloc[:1])
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

            logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
