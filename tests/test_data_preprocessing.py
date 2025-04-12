from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    create_time_dummies,
    load_data,
    preprocess_data,
    split_data,
    standardize_data,
)

np.random.seed(1001)
sample_df = pd.DataFrame(
    {
        "Timestamp": pd.date_range(start="2025-04-01 00:00:00", periods=100, freq="h"),
        "Feature1": np.random.randint(10, 100, size=100),
        "Feature2": np.random.randint(5, 95, size=100),
        "Hour": [i % 24 for i in range(100)],  # Repeat 0-23 to cover 100 hours
        "DayOfWeek": [i % 7 for i in range(100)],  # Repeat 0-6 to cover 100 hours
        "Month": [4] * 100,  # Repeat the month value 100 times
        "Target": np.random.randint(100, 1000, size=100),
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


def test_split_data_success():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df, target_column="Target", val_size=0.2)

    assert len(X_train) == 70
    assert len(X_val) == 20
    assert len(X_test) == 10
    assert len(y_train) == 70
    assert len(y_val) == 20
    assert len(y_test) == 10
    assert "Target" not in X_train.columns
    assert "Target" not in X_val.columns
    assert "Target" not in X_test.columns


def test_standardize_data_success():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df, target_column="Target", val_size=0.2)
    with patch("data_preprocessing.joblib.dump") as mock_dump:
        X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)
        assert X_train["Feature1"].mean() == pytest.approx(0, abs=0.5)
        assert X_train["Feature1"].std() == pytest.approx(1, abs=0.5)
        assert X_val["Feature1"].mean() == pytest.approx(0, abs=0.5)
        assert X_val["Feature1"].std() == pytest.approx(1, abs=0.5)
        assert X_test["Feature1"].mean() == pytest.approx(0, abs=0.5)
        assert X_test["Feature1"].std() == pytest.approx(1, abs=0.5)
        mock_dump.assert_called_once()


def test_create_time_dummies_success():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df, target_column="Target", val_size=0.2)
    X_train, X_val, X_test = create_time_dummies(X_train, X_val, X_test)

    assert "Hour" not in X_train.columns
    assert "DayOfWeek" not in X_train.columns
    assert "Month" not in X_train.columns
    assert "Hour_1" in X_train.columns
    assert "DayOfWeek_1" in X_train.columns
    assert "Month_4" not in X_train.columns


def test_preprocess_data_success():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df, target_column="Target", val_size=0.2)
    with patch("data_preprocessing.joblib.dump") as mock_dump:
        X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
        assert X_train["Feature1"].mean() == pytest.approx(0, abs=1e-6)
        assert X_train["Feature1"].std() == pytest.approx(1, abs=1e-2)  # Increased tolerance
        assert "Hour_1" in X_train.columns
        assert "DayOfWeek_1" in X_train.columns
        assert "Month_4" not in X_train.columns  # Dropped first category
        mock_dump.assert_called_once()
