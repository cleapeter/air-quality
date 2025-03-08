import mlflow
import pandas as pd
from data_preprocessing import load_data, preprocess_data, split_data
from train import train_model
from evaluate import evaluate_model
from config import load_config
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Air Quality: Random Forest Regressor")


def run_pipeline():
    # Load configuration
    config = load_config()

    # Get target column
    if "target_column" not in config["data"]:
        raise KeyError("Configuration error: 'target_column' not found in config['data']")
    target_column = config["data"]["target_column"]

    model_name = "random_forest_regressor"

    with mlflow.start_run():
        mlflow.log_param("target_column", target_column)

        # Load data
        df = load_data(config)

        # Process data
        df_scaled, df_target = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = split_data(df_scaled, df_target, target_column)

        # Train and evaluate model
        model_params = config["models"][model_name]
        model = train_model(X_train, y_train, model_name, model_params)
        metrics = evaluate_model(X_test, y_test, model)
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)

        input_example = pd.DataFrame(X_train.iloc[:1])
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)


if __name__ == "__main__":
    run_pipeline()
