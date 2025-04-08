from unittest.mock import patch

import pandas as pd
import pytest

from data_preprocessing import load_data, preprocess_data, split_data

sample_df = pd.DataFrame(
    {
        "Timestamp": pd.date_range(start="2025-04-01 00:00:00", periods=10, freq="h"),
        "Feature1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Feature2": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        "Hour": list(range(10)),
        "DayOfWeek": [1, 2, 3, 4, 5, 6, 0, 1, 2, 3],
        "Month": [4] * 10,
        "Target": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }
)


def test_load_data_success():
    config = {"data": {"clean_path": "dummy/path.csv"}}

    with patch("pandas.read_csv", return_value=sample_df) as mock_read_csv:
        df = load_data(config)
        mock_read_csv.assert_called_once_with("dummy/path.csv")
        assert df.index.name == "Timestamp"
        assert "Feature1" in df.columns
        assert "Feature2" in df.columns


def test_load_data_file_not_found():
    config = {"data": {"clean_path": "nonexistent.csv"}}

    with patch("pandas.read_csv", side_effect=FileNotFoundError("No such file or directory")):
        with pytest.raises(FileNotFoundError):
            load_data(config)


def test_preprocess_data_success():
    with patch("data_preprocessing.joblib.dump") as mock_dump:
        X, y = preprocess_data(sample_df.copy(), target_column="Target")
        assert "Target" not in X.columns
        assert y.name == "Target"
        assert mock_dump.called


def test_preprocess_data_no_numeric_features():
    df = sample_df.drop(columns=["Feature1", "Feature2"])
    with pytest.raises(ValueError):
        preprocess_data(df, target_column="Target")


def test_split_data_creates_files():
    X = sample_df.drop(columns=["Target"])
    y = sample_df["Target"]

    with (
        patch("data_preprocessing.os.makedirs") as mock_makedirs,
        patch("data_preprocessing.pd.DataFrame.to_csv") as mock_df_to_csv,
        patch("data_preprocessing.pd.Series.to_csv") as mock_series_to_csv,
    ):
        X_train, X_test, y_train, y_test = split_data(X, y, target_name="Target")

        total_calls = mock_df_to_csv.call_count + mock_series_to_csv.call_count
        assert total_calls == 4

        mock_makedirs.assert_called_once_with("data/processed/split/Target", exist_ok=True)

        assert not X_train.empty
        assert not X_test.empty
        assert not y_train.empty
        assert not y_test.empty
