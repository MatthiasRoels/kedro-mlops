from contextlib import contextmanager

import polars as pl
import pytest
from kedro_mlops.library.preprocessing.kbins_discretizer import KBinsDiscretizer
from sklearn.exceptions import NotFittedError


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "strategy, expectation",
    [("trees", pytest.raises(ValueError)), ("quantile", does_not_raise())],
)
def test_fit_exception(strategy, expectation):
    with expectation:
        _ = KBinsDiscretizer(strategy=strategy)


@pytest.mark.parametrize(
    "use_lazy_frame, auto_adapt_bins",
    [(False, False), (False, True), (True, True)],
    ids=["eager_default", "eager_auto_adapt_bins", "lazy_auto_adapt_bins"],
)
def test_fit(use_lazy_frame: bool, auto_adapt_bins: bool):
    data = pl.DataFrame({"variable": [*list(range(10)), *([None] * 5)]})

    if use_lazy_frame:
        data = data.lazy()

    discretizer = KBinsDiscretizer(
        n_bins=3,
        strategy="uniform",
        auto_adapt_bins=auto_adapt_bins,
    )

    discretizer.fit(data, ["variable"])

    actual_bin_edges = discretizer.bin_edges_by_column_["variable"]
    expected_bin_edges = [3.0, 6.0]
    if auto_adapt_bins:
        expected_bin_edges = [4.5]

    assert actual_bin_edges == expected_bin_edges

    actual_bin_labels = discretizer.bin_labels_by_column_["variable"]
    expected_bin_labels = ["(-inf, 3.0]", "(3.0, 6.0]", "(6.0, inf)"]
    if auto_adapt_bins:
        expected_bin_labels = ["(-inf, 4.5]", "(4.5, inf)"]

    assert actual_bin_labels == expected_bin_labels


def test_transform_when_not_fitted():
    discretizer = KBinsDiscretizer()
    with pytest.raises(NotFittedError):
        discretizer.transform(data=pl.DataFrame())


@pytest.mark.parametrize(
    "use_lazy_frame",
    [(False,), (True,)],
    ids=["eager", "lazy"],
)
def test_transform(use_lazy_frame: bool):
    data = pl.DataFrame({"variable": list(range(10))})

    if use_lazy_frame:
        data = data.lazy()

    discretizer = KBinsDiscretizer(n_bins=2, strategy="uniform")

    discretizer.fit(data, ["variable"])

    tf_data = discretizer.transform(data)

    assert tf_data.dtypes == [pl.Categorical]

    if use_lazy_frame:
        tf_data = tf_data.collect()

    actual = tf_data.to_dict(as_series=False)["variable"]
    expected = [*(["(-inf, 4.5]"] * 5), *(["(4.5, inf)"] * 5)]

    assert actual == expected


@pytest.mark.parametrize(
    "use_lazy_frame",
    [(False,), (True,)],
    ids=["eager", "lazy"],
)
def test_transform_fewer_columns(use_lazy_frame: bool):
    data = pl.DataFrame({"feat": list(range(10)), "excluded_feat": list(range(10))})

    if use_lazy_frame:
        data = data.lazy()

    discretizer = KBinsDiscretizer(n_bins=2, strategy="uniform")

    discretizer.fit(data, ["feat", "excluded_feat"])

    tf_data = discretizer.transform(data.select(["feat"]))

    assert tf_data.dtypes == [pl.Categorical]

    if use_lazy_frame:
        tf_data = tf_data.collect()

    actual = tf_data.to_dict(as_series=False)["feat"]
    expected = [*(["(-inf, 4.5]"] * 5), *(["(4.5, inf)"] * 5)]

    assert actual == expected


def test_fit_transform():
    data = pl.DataFrame({"feat": list(range(10))})

    discretizer = KBinsDiscretizer(n_bins=2, strategy="uniform")

    tf_data = discretizer.fit_transform(data, ["feat"])

    assert tf_data.dtypes == [pl.Categorical]

    actual = tf_data.to_dict(as_series=False)["feat"]
    expected = [*(["(-inf, 4.5]"] * 5), *(["(4.5, inf)"] * 5)]

    assert actual == expected


@pytest.mark.parametrize(
    "use_lazy_frame, include_missing",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["eager_no_missing", "eager_missing", "lazy_no_missing", "lazy_missing"],
)
def test_fit_with_uniform_strategy(use_lazy_frame: bool, include_missing: bool):
    if include_missing:
        data = pl.DataFrame({"variable": list(range(10))})
    else:
        data = pl.DataFrame({"variable": [*list(range(10)), None]})

    if use_lazy_frame:
        data = data.lazy()

    n_bins_by_column = {"variable": 3}

    discretizer = KBinsDiscretizer(n_bins=3, strategy="uniform")

    discretizer._fit_with_uniform_strategy(data, n_bins_by_column)

    actual_bin_edges = discretizer.bin_edges_by_column_["variable"]
    expected_bin_edges = [3.0, 6.0]

    assert actual_bin_edges == expected_bin_edges

    actual_bin_labels = discretizer.bin_labels_by_column_["variable"]
    expected_bin_labels = ["(-inf, 3.0]", "(3.0, 6.0]", "(6.0, inf)"]

    assert actual_bin_labels == expected_bin_labels


@pytest.mark.parametrize(
    "use_lazy_frame, include_missing",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["eager_no_missing", "eager_missing", "lazy_no_missing", "lazy_missing"],
)
def test_fit_with_quantile_strategy(use_lazy_frame: bool, include_missing: bool):
    if include_missing:
        data = pl.DataFrame({"variable": list(range(11))})
    else:
        data = pl.DataFrame({"variable": [*list(range(11)), None]})

    if use_lazy_frame:
        data = data.lazy()

    n_bins_by_column = {"variable": 4}

    discretizer = KBinsDiscretizer(n_bins=4, strategy="quantile", starting_precision=1)

    discretizer._fit_with_quantile_strategy(data, n_bins_by_column)

    actual_bin_edges = discretizer.bin_edges_by_column_["variable"]
    expected_bin_edges = [2.5, 5.0, 7.5]

    assert actual_bin_edges == expected_bin_edges

    actual_bin_labels = discretizer.bin_labels_by_column_["variable"]
    expected_bin_labels = ["(-inf, 2.5]", "(2.5, 5.0]", "(5.0, 7.5]", "(7.5, inf)"]

    assert actual_bin_labels == expected_bin_labels


@pytest.mark.parametrize(
    "n_bins, expectation",
    [
        (1, pytest.raises(ValueError)),
        (10.5, pytest.raises(ValueError)),
        (2, does_not_raise()),
    ],
)
def test_validate_n_bins_exception(n_bins, expectation):
    with expectation:
        assert KBinsDiscretizer()._validate_n_bins(n_bins=n_bins) is None


@pytest.mark.parametrize(
    "bin_edges, starting_precision, expected",
    [
        ([-10, 0, 1, 2], 1, 1),
        ([-10, 0, 1, 1.01], 0, 2),
        ([-10, 0, 1, 1.1], 1, 1),
        ([-10, 0, 1, 2], -1, 0),
        ([-10, 0, 10, 21], -1, -1),
    ],
    ids=[
        "less precision",
        "more precision",
        "equal precision",
        "negative start",
        "round up",
    ],
)
def test_compute_minimal_precision_of_bin_edges(
    bin_edges, starting_precision, expected
):
    discretizer = KBinsDiscretizer(starting_precision=starting_precision)

    actual = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)

    assert actual == expected


@pytest.mark.parametrize(
    "left_closed, label_format, bin_edges, expected",
    [
        (False, None, [1, 2, 3], ["(-inf, 1]", "(1, 2]", "(2, 3]", "(3, inf)"]),
        (False, "{} - {}", [1, 2, 3], ["<= 1", "1 - 2", "2 - 3", "> 3"]),
        (True, None, [1, 2, 3], ["(-inf, 1)", "[1, 2)", "[2, 3)", "[3, inf)"]),
        (True, "{} - {}", [1, 2, 3], ["< 1", "1 - 2", "2 - 3", ">= 3"]),
    ],
    ids=["defaults", "other_label_fmt", "left_closed", "left_closed_other_label_fmt"],
)
def test_create_bin_labels_from_edges(
    left_closed: bool, label_format: str, bin_edges: list, expected: list
):
    discretizer = KBinsDiscretizer(left_closed=left_closed, label_format=label_format)

    actual = discretizer._create_bin_labels_from_edges(bin_edges)

    assert actual == expected
