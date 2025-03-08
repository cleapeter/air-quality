from data_preprocessing import load_data, preprocess_data, split_data
from train import train_model
from evaluate import evaluate_model
from config import load_config


def run_pipeline():
    # Load configuration
    config = load_config()

    # Load data
    df = load_data(config)

    # Get target column
    if "target_column" not in config["data"]:
        raise KeyError("Configuration error: 'target_column' not found in config['data']")
    target_column = config["data"]["target_column"]

    # Process data
    df_scaled, df_target = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = split_data(df_scaled, df_target, target_column)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    mse, r2, mae = evaluate_model(X_test, y_test, model)

    print(mse, r2, mae)


if __name__ == "__main__":
    run_pipeline()
