import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_model(X, y, model_name, model_params):
    if model_name == "linear_regression":
        model = LinearRegression(**model_params)
    elif model_name == "random_forest_regressor":
        model = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X, y)

    # Save the model
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)

    return model
