import joblib
from sklearn.linear_model import LinearRegression


def train_model(X, y):
    # Drop columns not used for training
    X = X.drop(columns=["Timestamp"])

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, "models/linear_regression.pkl")

    return model
