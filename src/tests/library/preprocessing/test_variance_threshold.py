import polars as pl
import pytest
from kedro_mlops.library.preprocessing.variance_threshold import VarianceThreshold
from polars.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError


@pytest.fixture()
def data():
    return pl.DataFrame(
        {
            "feat1": list(range(100)),
            "feat2": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * 10,
            "const1": [0] * 100,
            "const2": ["a"] * 100,
        }
    )


@pytest.mark.parametrize(
    "use_lazy",
    [False, True],
    ids=["eager", "lazy"],
)
def test_fit(data, use_lazy: bool):
    if use_lazy:
        data = data.lazy()

    variance_threshold = VarianceThreshold()
    variance_threshold.fit(data)

    actual = variance_threshold.columns_to_drop_
    expected = ["const1", "const2"]

    assert actual == expected


@pytest.mark.parametrize(
    "use_lazy",
    [False, True],
    ids=["eager", "lazy"],
)
def test_transform(data, use_lazy: bool):
    if use_lazy:
        data = data.lazy()

    variance_threshold = VarianceThreshold()
    variance_threshold.fit(data)

    actual = variance_threshold.transform(data)
    expected = data.select(["feat1", "feat2"])

    if use_lazy:
        actual = actual.collect()
        expected = expected.collect()

    assert_frame_equal(actual, expected)


def test_transform_without_fit(data):
    variance_threshold = VarianceThreshold()
    with pytest.raises(NotFittedError):
        variance_threshold.transform(data)


def test_fit_transform(data):
    variance_threshold = VarianceThreshold()
    actual = variance_threshold.fit_transform(data)
    expected = data.select(["feat1", "feat2"])

    assert_frame_equal(actual, expected)
