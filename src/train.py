import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def train_model(X, y, model_name, model_params):
    try:
        logger.info(f"Starting training for model: {model_name}")

        if model_name == "linear_regression":
            model = LinearRegression(**model_params)
        elif model_name == "random_forest_regressor":
            model = RandomForestRegressor(**model_params)
        else:
            logger.error(f"Unsupported model: {model_name}")
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X, y)
        logger.info(f"Training completed for model: {model_name}")

        return model
    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        raise
