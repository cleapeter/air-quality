import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(X, y, timestamps, model):
    # Make predictions
    y_pred = model.predict(X)

    # Compute regression metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    metrics = {"mse": mse, "r2": r2, "mae": mae}

    output_dir = "./data/output"
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({"timestamp": timestamps, "y_true": y, "y_pred": y_pred})
    predictions_filename = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_filename, index=False)

    # Scatter plot comparing predicted vs actual values with a diagonal line
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=predictions_df["y_true"], y=predictions_df["y_pred"], color="blue", label="Predicted vs True")

    # Add diagonal line representing perfect predictions
    max_val = max(predictions_df["y_true"].max(), predictions_df["y_pred"].max())
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect Prediction Line")

    # Add grid and labels
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Ground Truth (Validation Set)")
    plt.grid(True)
    plt.legend()

    scatter_plot_filename = os.path.join(output_dir, "predicted_vs_ground_truth.png")
    plt.savefig(scatter_plot_filename)
    plt.close()

    # Line plot of true vs predicted values over timestamps
    plt.figure(figsize=(10, 6))
    plt.plot(
        predictions_df["timestamp"],
        predictions_df["y_true"],
        label="True Values",
        color="blue",
        linestyle="-",
        marker="o",
        markersize=3,
    )
    plt.plot(
        predictions_df["timestamp"],
        predictions_df["y_pred"],
        label="Predicted Values",
        color="red",
        linestyle="-",
        marker="x",
        markersize=3,
    )

    subset_timestamps = predictions_df["timestamp"][::48]  # Show every 10th timestamp for clarity
    plt.xticks(subset_timestamps, rotation=90)  # Rotate x-axis labels for better readability

    # Rotate the x-axis labels for better readability if timestamps are long
    plt.xlabel("Timestamp")
    plt.ylabel("Values")
    plt.title("True vs Predicted Values Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    time_series_plot_filename = os.path.join(output_dir, "true_vs_predicted_over_time.png")
    plt.savefig(time_series_plot_filename)
    plt.close()

    return metrics, predictions_filename, scatter_plot_filename, time_series_plot_filename
