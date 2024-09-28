import pandas as pd
import pytest
import hydra

with hydra.initialize(config_path="..", version_base="1.1"):
    conf = hydra.compose(config_name="config")

cloned_categorical_features = conf['main']['category_features']


@pytest.fixture(scope="module")
def cleaned_data():
    return pd.read_csv("data/census_clean.csv", skipinitialspace=True)


def test_salary_range(cleaned_data):
    """
    Tests that the salary column has the values in expected range.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "<=50K",
        ">50K"
    }

    assert set(cleaned_data["salary"]) == expected_values


def test_column_ranges(cleaned_data):
    """
    Test for check if some column's values is in range

    Args:
        data (pd.DataFrame): Dataset for testing
    """

    ranges = {
        "age": (17, 90),
        "education-num": (1, 16),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    }

    for col_name, (minimum, maximum) in ranges.items():
        assert cleaned_data[col_name].min() >= minimum
        assert cleaned_data[col_name].max() <= maximum
