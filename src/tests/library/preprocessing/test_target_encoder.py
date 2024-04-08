import polars as pl
import pytest
from kedro_mlops.library.preprocessing.target_encoder import TargetEncoder
from polars.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError


@pytest.fixture(scope="module")
def feat_values_binary_cls() -> list:
    return [
        "positive",
        "positive",
        "negative",
        "neutral",
        "negative",
        "positive",
        "negative",
        "neutral",
        "neutral",
        "neutral",
    ]


@pytest.fixture(scope="module")
def target_binary_cls() -> list:
    return [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]


@pytest.fixture(scope="module")
def encoded_feat_values_binary_cls() -> list:
    return [
        0.666667,
        0.666667,
        0.333333,
        0.50000,
        0.333333,
        0.666667,
        0.333333,
        0.50000,
        0.50000,
        0.50000,
    ]


@pytest.fixture(scope="module")
def feat_values_regression() -> list:
    return [
        "positive",
        "positive",
        "negative",
        "neutral",
        "negative",
        "positive",
        "negative",
        "neutral",
        "neutral",
        "neutral",
        "positive",
    ]


@pytest.fixture(scope="module")
def target_regression() -> list:
    return [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]


@pytest.fixture(scope="module")
def encoded_feat_values_regression() -> list:
    return [
        4.500000,
        4.500000,
        -4.666667,
        0.250000,
        -4.666667,
        4.500000,
        -4.666667,
        0.250000,
        0.250000,
        0.250000,
        4.500000,
    ]


def test_target_encoder_constructor_weight_value_error():
    with pytest.raises(ValueError):
        TargetEncoder([], weight=-1)


def test_target_encoder_constructor_imputation_value_error():
    with pytest.raises(ValueError):
        TargetEncoder([], imputation_strategy="median")


@pytest.mark.parametrize(
    "variable, target, expected, use_lazy_api",
    [
        (
            "feat_values_binary_cls",
            "target_binary_cls",
            {"negative": 0.333333, "neutral": 0.50000, "positive": 0.666667},
            True,
        ),
        (
            "feat_values_binary_cls",
            "target_binary_cls",
            {"negative": 0.333333, "neutral": 0.50000, "positive": 0.666667},
            False,
        ),
        (
            "feat_values_regression",
            "target_regression",
            {"negative": -4.666667, "neutral": 0.250000, "positive": 4.500000},
            True,
        ),
        (
            "feat_values_regression",
            "target_regression",
            {"negative": -4.666667, "neutral": 0.250000, "positive": 4.500000},
            False,
        ),
    ],
    ids=[
        "binary_classification_lazy",
        "binary_classification_eager",
        "linear_regression_lazy",
        "linear_regression_eager",
    ],
)
def test_target_encoder_fit(
    variable: list, target: list, expected: dict, use_lazy_api: bool, request
):
    """Test fit method of TargetEncoder.

    Testing is performed on both binary classification and linear regression
    """
    df = pl.DataFrame(
        {
            "variable": request.getfixturevalue(variable),
            "target": request.getfixturevalue(target),
        }
    )

    if use_lazy_api:
        df = df.lazy()

    encoder = TargetEncoder(column_names=["variable"])
    encoder.fit(X=df.select("variable"), y=df.select("target"))

    actual = encoder.mapping_["variable"]

    assert actual == pytest.approx(expected, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("use_lazy_api", [False, True], ids=["eager", "lazy"])
def test_target_encoder_fit_multiple_columns_different_dtypes(
    feat_values_binary_cls, target_binary_cls, use_lazy_api: bool
):
    df = pl.DataFrame(
        {
            "feat_1": feat_values_binary_cls,
            "feat_2": [0] * 5 + [1, 1, 2, 1, 2],
            "target": target_binary_cls,
        }
    )

    if use_lazy_api:
        df = df.lazy()

    encoder = TargetEncoder(column_names=["feat_1", "feat_2"])
    encoder.fit(X=df.select(["feat_1", "feat_2"]), y=df.select("target"))

    actual = encoder.mapping_
    expected = {
        "feat_1": {"negative": 0.333333, "neutral": 0.50000, "positive": 0.666667},
        "feat_2": {0: 0.6, 1: 0.3333, 2: 0.5},
    }

    assert actual["feat_1"] == pytest.approx(expected["feat_1"], rel=1e-3, abs=1e-3)
    assert actual["feat_2"] == pytest.approx(expected["feat_2"], rel=1e-3, abs=1e-3)


def test_target_encoder_transform_when_not_fitted():
    df = pl.DataFrame()

    encoder = TargetEncoder([])
    with pytest.raises(NotFittedError):
        encoder.transform(X=df)


@pytest.mark.parametrize(
    "variable, target, variable_enc, use_lazy_api",
    [
        (
            "feat_values_binary_cls",
            "target_binary_cls",
            "encoded_feat_values_binary_cls",
            True,
        ),
        (
            "feat_values_binary_cls",
            "target_binary_cls",
            "encoded_feat_values_binary_cls",
            False,
        ),
        (
            "feat_values_regression",
            "target_regression",
            "encoded_feat_values_regression",
            True,
        ),
        (
            "feat_values_regression",
            "target_regression",
            "encoded_feat_values_regression",
            False,
        ),
    ],
    ids=[
        "binary_classification_lazy",
        "binary_classification_eager",
        "linear_regression_lazy",
        "linear_regression_eager",
    ],
)
def test_target_encoder_transform(
    variable: list, target: list, variable_enc: dict, use_lazy_api: bool, request
):
    """Test transform method of TargetEnconder.

    Testing is performed on both binary classification and linear regression
    """
    df = pl.DataFrame(
        {
            "variable": request.getfixturevalue(variable),
            "target": request.getfixturevalue(target),
        }
    )

    if use_lazy_api:
        df = df.lazy()

    encoder = TargetEncoder(column_names=["variable"])
    encoder.fit(X=df.select("variable"), y=df.select("target"))

    actual = encoder.transform(X=df.select("variable"))
    expected = df.with_columns(
        pl.Series(name="variable_enc", values=request.getfixturevalue(variable_enc))
    )
    if use_lazy_api:
        actual = actual.collect()
        expected = expected.collect()

    assert_frame_equal(actual, expected.select(pl.exclude("target")))


@pytest.mark.parametrize(
    "variable, target, variable_enc, additional_enc_value",
    [
        (
            "feat_values_binary_cls",
            "target_binary_cls",
            "encoded_feat_values_binary_cls",
            0.333333,
        ),
        (
            "feat_values_regression",
            "target_regression",
            "encoded_feat_values_regression",
            -4.666667,
        ),
    ],
    ids=["binary_classification", "linear_regression"],
)
def test_target_encoder_transform_new_category(
    variable: list,
    target: list,
    variable_enc: list,
    additional_enc_value: float,
    request,
):
    df = pl.DataFrame(
        {
            "variable": request.getfixturevalue(variable),
            "target": request.getfixturevalue(target),
        }
    )

    encoder = TargetEncoder(column_names=["variable"], imputation_strategy="min")
    encoder.fit(X=df.select("variable"), y=df.select("target"))

    df_extended = df.extend(pl.DataFrame({"variable": "new", "target": 1}))

    encoded_values = [*request.getfixturevalue(variable_enc), additional_enc_value]

    actual = encoder.transform(X=df_extended.select("variable"))
    expected = df_extended.with_columns(
        pl.Series(name="variable_enc", values=encoded_values)
    )

    assert_frame_equal(actual, expected.select(pl.exclude("target")))


def test_target_encoder_transform_fewer_columns(
    feat_values_binary_cls, target_binary_cls, encoded_feat_values_binary_cls
):
    """Test transform function, but exclude a column compared to calling fit.
    This is to ensure we can do not have to refit the estimator after feature selection.
    """
    df = pl.DataFrame(
        {
            "feat": feat_values_binary_cls,
            "excluded_variable": feat_values_binary_cls,
            "target": target_binary_cls,
        }
    )

    encoder = TargetEncoder(column_names=["feat", "excluded_variable"])
    encoder.fit(X=df.select(["feat", "excluded_variable"]), y=df.select("target"))
    actual = encoder.transform(X=df.select("feat"))

    expected = df.with_columns(
        pl.Series(name="feat_enc", values=encoded_feat_values_binary_cls)
    ).select(["feat", "feat_enc"])

    assert_frame_equal(actual, expected)


def test_target_encoder_fit_transform(
    feat_values_binary_cls, target_binary_cls, encoded_feat_values_binary_cls
):
    df = pl.DataFrame(
        {
            "variable": feat_values_binary_cls,
            "target": target_binary_cls,
        }
    )

    encoder = TargetEncoder(column_names=["variable"])
    actual = encoder.fit_transform(X=df.select("variable"), y=df.select("target"))

    expected = df.with_columns(
        pl.Series(name="variable_enc", values=encoded_feat_values_binary_cls)
    )

    assert_frame_equal(actual, expected.select(pl.exclude("target")))


@pytest.mark.parametrize(
    "cname, expected",
    [
        ("test_column", "test_column_enc"),
        ("test_column_bin", "test_column_enc"),
        ("test_column_processed", "test_column_enc"),
        ("test_column_cleaned", "test_column_enc"),
    ],
)
def test_target_encoder_clean_column_name(cname, expected):
    encoder = TargetEncoder([])
    actual = encoder._clean_column_name(cname)

    assert actual == expected
