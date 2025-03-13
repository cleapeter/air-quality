import os

from data_preprocessing import load_data


def test_load_data_from_csv():
    test_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "testAirQuality.csv")

    df = load_data({"data": {"clean_path": test_csv_path}})

    assert df.shape == (2, 13)
    assert df.index.name == "Timestamp"
    assert "CO_reference_mg_per_m3" in df.columns
