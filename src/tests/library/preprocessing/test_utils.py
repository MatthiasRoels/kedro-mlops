import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kedro_mlops.library.preprocessing.utils import (
    stratified_train_test_split_binary_target,
    train_test_split_continuous_target,
    univariate_feature_selection_classification,
    univariate_feature_selection_regression,
)


@pytest.fixture(scope="module")
def data():
    return pl.DataFrame({"variable": ["a"] * 100, "target": [0] * 80 + [1] * 20})


@pytest.fixture(scope="module")
def features_binary_target():
    return {
        "target": [0, 0, 0, 1, 1],
        "good_feature": [0.2, 0.3, 0.7, 0.6, 0.9],
        "bad_feature": [0.8, 0.6, 0.7, 0.3, 0.9],
    }


@pytest.fixture(scope="module")
def features_continuous_target():
    return {
        "target": [1, 2, 3, 4, 5],
        "good_feature": [1.2, 1.8, 2.7, 4.3, 5.2],
        "bad_feature": [0.5, 3, 2.6, 2.9, 7],
    }


@pytest.fixture(scope="module")
def sampled_data(data):
    return data.sample(fraction=1.0, seed=42)


@pytest.mark.parametrize("use_lazy", [False, True], ids=["eager", "lazy"])
def test_stratified_train_test_split_binary_target(data, use_lazy: bool):
    if use_lazy:
        data = data.lazy()

    expected_values = ["test"] * 24 + ["train"] * 56 + ["test"] * 6 + ["train"] * 14

    actual = stratified_train_test_split_binary_target(data, "target", 0.3)
    expected = data.with_columns(pl.Series(name="split", values=expected_values))

    if use_lazy:
        actual = actual.collect()
        expected = expected.collect()

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "use_lazy, shuffle",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["eager", "eager_shuffled", "lazy", "lazy_shuffled"],
)
def test_stratified_train_test_split_binary_target_sampled_data(
    sampled_data, use_lazy: bool, shuffle: bool
):
    if use_lazy:
        sampled_data = sampled_data.lazy()

    splitted_data = stratified_train_test_split_binary_target(
        sampled_data, "target", test_size=0.3, shuffle_data=shuffle, random_seed=42
    )

    actual_raw = splitted_data.group_by(["split", "target"]).len()
    if use_lazy:
        actual_raw = actual_raw.collect()

    actual = actual_raw.sort(["split", "target"]).to_dict(as_series=False)
    expected = {
        "split": ["test", "test", "train", "train"],
        "target": [0, 1, 0, 1],
        "len": [24, 6, 56, 14],
    }
    assert actual == expected


@pytest.mark.parametrize(
    "use_lazy, shuffle",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["eager", "eager_shuffled", "lazy", "lazy_shuffled"],
)
def test_train_test_split_continuous_target(
    sampled_data, use_lazy: bool, shuffle: bool
):
    if use_lazy:
        sampled_data = sampled_data.lazy()

    splitted_data = train_test_split_continuous_target(
        sampled_data, test_size=0.3, shuffle_data=shuffle, random_seed=42
    )

    actual_raw = splitted_data.group_by("split").len()
    if use_lazy:
        actual_raw = actual_raw.collect()

    actual = actual_raw.sort(["split"]).to_dict(as_series=False)
    expected = {
        "split": ["test", "train"],
        "len": [30, 70],
    }
    assert actual == expected


@pytest.mark.parametrize("use_lazy_api", [False, True], ids=["eager", "lazy"])
def test_univariate_feature_selection_classification(
    features_binary_target, use_lazy_api: bool
):
    data = pl.DataFrame(features_binary_target)
    if use_lazy_api:
        data = data.lazy()

    actual = univariate_feature_selection_classification(data, "target")
    expected = data.select(["target", "good_feature"])

    if use_lazy_api:
        actual = actual.collect()
        expected = expected.collect()

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("use_lazy_api", [False, True], ids=["eager", "lazy"])
def test_univariate_feature_selection_regression(
    features_continuous_target, use_lazy_api: bool
):
    data = pl.DataFrame(features_continuous_target)
    if use_lazy_api:
        data = data.lazy()

    actual = univariate_feature_selection_regression(data, "target", threshold=1)
    expected = data.select(["target", "good_feature"])

    if use_lazy_api:
        actual = actual.collect()
        expected = expected.collect()

    assert_frame_equal(actual, expected)
