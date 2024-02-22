import polars as pl
from sklearn.base import BaseEstimator

from .kbins_discretizer import KBinsDiscretizer
from .target_encoder import TargetEncoder
from .variance_threshold import VarianceThreshold


def stratified_train_test_split_binary_target(
    data: pl.DataFrame | pl.LazyFrame,
    target: str,
    test_size: float,
    shuffle_data: bool = False,
    random_seed: int | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Adds a `split` column with train/test values to the dataset when the target
    is binary. It uses the stratified approach which means the incidence in train/test
    are the same as the original dataset.

    Train set = data on which the model is trained and on which the encoding is based.
    Test set = data that generates the final performance metrics.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Input dataset to split into train-test sets.
    test_size : float, optional
        Percentage of data to put in test set.
    shuffle_data : float, optional
        Whether or not to shuffle the data before splitting
    random_seed : float, optional
        Seed for the random number generator. If set to None (default),
        a random seed is generated for each sample operation.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with additional split column.
    """
    cnames = data.columns
    example_col = cnames[0] if cnames[0] != target else cnames[1]
    if shuffle_data:
        if isinstance(data, pl.LazyFrame):
            df = data.collect().sample(fraction=1.0, seed=random_seed).lazy()
        else:
            df = data.sample(fraction=1.0, seed=random_seed)
    else:
        df = data.select(pl.all())

    res = df.group_by(target).len()

    if isinstance(res, pl.LazyFrame):
        res = res.collect()

    row_counts_raw = res.to_dict(as_series=False)
    row_counts = {
        str(k): v for k, v in zip(row_counts_raw[target], row_counts_raw["len"])
    }

    # Add a row_number partitioned by the target and use it to compute the splits
    # We can then use the row number to select the desired number of samples for the
    # test set with the correct proportion of the target
    return (
        df.with_columns(
            row_number=pl.col(example_col).rank(method="ordinal").over(target)
        )
        .with_columns(
            split=pl.when(
                pl.col(target) == 1,
                pl.col("row_number") <= row_counts["1"] * test_size,
            )
            .then(pl.lit("test"))
            .when(
                pl.col(target) == 0,
                pl.col("row_number") <= row_counts["0"] * test_size,
            )
            .then(pl.lit("test"))
            .otherwise(pl.lit("train"))
        )
        .select([*cnames, "split"])
    )


def apply_variance_threshold(  # pragma: no cover
    data: pl.DataFrame | pl.LazyFrame,
    threshold: float = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Wrapper function around VarianceThreshold

    Parameters
    ----------
    data : pl.LazyFrame | pl.DataFrame
        Data on which a feature pre-selection is performed

    Returns
    -------
    pl.LazyFrame | pl.DataFrame
        data with only selected features
    """
    variance_threshold = VarianceThreshold(threshold)
    return variance_threshold.fit_transform(data)


def fit_discretizer(  # pragma: no cover
    data: pl.DataFrame | pl.LazyFrame,
    column_names: list,
    discretizer_config: dict,
) -> KBinsDiscretizer:
    """Wrapper around KBinsDiscretizer.fit

    Parameters
    ----------
    data : pl.LazyFrame | pl.DataFrame
        Data to fit the estimator on
    column_names: list
        Names of the columns of the DataFrame suitable for discretization
    discretizer_config: dict
        input parameter for class constructor

    Returns
    -------
    KBinsDiscretizer
        Fitted estimator

    """
    discretizer = KBinsDiscretizer(**discretizer_config)
    discretizer.fit(data.filter(pl.col("split") == "train"), column_names)

    return discretizer


def fit_encoder(  # pragma: no cover
    data: pl.DataFrame | pl.LazyFrame,
    column_names: list,
    target_column: str,
    encoder_config: dict,
) -> TargetEncoder:
    """Wrapper around TargetEncoder.fit
    Parameters
    ----------
    data : pl.LazyFrame | pl.DataFrame
        Data to fit the estimator on
    column_names: list
        Names of the columns of the DataFrame to be encoded
    target_column : str
            Column name of the target.
    encoder_config: dict
        input parameter for class constructor

    Returns
    -------
    TargetEncoder
        Fitted estimator
    """
    encoder = TargetEncoder(**encoder_config)
    encoder.fit(data.filter(pl.col("split") == "train"), column_names, target_column)

    return encoder


def transform_data(  # pragma: no cover
    data: pl.DataFrame | pl.LazyFrame,
    fitted_estimator: BaseEstimator,
) -> pl.DataFrame | pl.LazyFrame:
    """Wrapper around transform methods of estimators"""
    return fitted_estimator.transform(data)
