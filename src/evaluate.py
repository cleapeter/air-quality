from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(X, y, model):
    # Remove non-feature columns
    X = X.drop(columns=["Timestamp"])

    # Make predictions
    y_pred = model.predict(X)

    # Compute regression metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return mse, r2, mae
