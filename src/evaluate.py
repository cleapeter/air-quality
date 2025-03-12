from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(X, y, model):
    # Make predictions
    y_pred = model.predict(X)

    # Compute regression metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    metrics = {"mse": mse, "r2": r2, "mae": mae}

    return metrics
