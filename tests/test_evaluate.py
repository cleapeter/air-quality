import numpy as np
from sklearn.ensemble import RandomForestRegressor

from evaluate import evaluate_model


def test_evaluate_model():
    # Mock data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # Train a simple model
    model = RandomForestRegressor(n_estimators=1, random_state=42)
    model.fit(X, y)

    # Evaluate
    metrics = evaluate_model(X, y, model)

    assert "mse" in metrics
    assert "r2" in metrics
    assert "mae" in metrics
    assert isinstance(metrics["mse"], float)
    assert isinstance(metrics["r2"], float)
    assert isinstance(metrics["mae"], float)
