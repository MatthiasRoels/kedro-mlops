import numpy as np
import polars as pl
import pytest
from kedro_mlops.library.model_building.linear import (
    get_predictions,
    sequential_feature_selection,
    train_model,
)
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.from_pandas(load_breast_cancer(as_frame=True)["frame"])


@pytest.fixture(scope="module")
def mod_params() -> dict:
    return {
        "model": {
            "class": "sklearn.linear_model.LogisticRegression",
            "kwargs": {
                "fit_intercept": True,
                "C": 1e9,
                "solver": "liblinear",
                "random_state": 42,
            },
        },
        "sequential_feature_selection_kwargs": {
            "tol": 10e-3,
            "direction": "forward",
            "scoring": "roc_auc",
            "cv": 5,
        },
    }


def test_sequential_feature_selection(data, mod_params):
    actual = sequential_feature_selection(data, "target", mod_params)
    expected = ["worst perimeter"]

    assert list(actual) == expected


def test_train_model(data, mod_params, mocker):
    def mock_fit(self, X, y):
        return None

    mocker.patch(
        "sklearn.linear_model.LogisticRegression.fit",
        mock_fit,
    )

    selected_features = ["worst perimeter"]
    actual = train_model(data, "target", selected_features, mod_params)

    assert isinstance(actual, LogisticRegression)


@pytest.mark.parametrize("use_lazy_api", [False, True], ids=["eager", "lazy"])
def test_get_predictions(data, use_lazy_api: bool, mocker):
    if use_lazy_api:
        data = data.lazy()

    def mock_predict_proba(self, X):
        return np.random.default_rng().uniform(0, 1, (569, 2))

    mocker.patch(
        "sklearn.linear_model.LogisticRegression.predict_proba",
        mock_predict_proba,
    )

    logit = LogisticRegression()
    selected_features = ["worst perimeter"]
    actual = get_predictions(data, selected_features, logit)

    assert "predictions" in actual.columns
