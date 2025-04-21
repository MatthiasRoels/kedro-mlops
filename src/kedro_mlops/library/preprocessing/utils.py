import logging

import polars as pl
import polars_ds as pld  # noqa: F401

logger = logging.getLogger(__name__)


def stratified_train_test_split_binary_target(
    data: pl.DataFrame | pl.LazyFrame,
    target_column: str,
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
    target_column: str
        Column name of the target.
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
    example_col = cnames[0] if cnames[0] != target_column else cnames[1]
    if shuffle_data:
        if isinstance(data, pl.LazyFrame):
            df = data.collect().sample(fraction=1.0, seed=random_seed).lazy()
        else:
            df = data.sample(fraction=1.0, seed=random_seed)
    else:
        df = data.select(pl.all())

    res = df.group_by(target_column).len()

    if isinstance(res, pl.LazyFrame):
        res = res.collect()

    row_counts_raw = res.to_dict(as_series=False)
    row_counts = {
        str(k): v
        for k, v in zip(
            row_counts_raw[target_column], row_counts_raw["len"], strict=False
        )
    }

    # Add a row_number partitioned by the target and use it to compute the splits
    # We can then use the row number to select the desired number of samples for the
    # test set with the correct proportion of the target
    return (
        df.with_columns(
            row_number=pl.col(example_col).rank(method="ordinal").over(target_column)
        )
        .with_columns(
            split=pl.when(
                pl.col(target_column) == 1,
                pl.col("row_number") <= row_counts["1"] * test_size,
            )
            .then(pl.lit("test"))
            .when(
                pl.col(target_column) == 0,
                pl.col("row_number") <= row_counts["0"] * test_size,
            )
            .then(pl.lit("test"))
            .otherwise(pl.lit("train"))
        )
        .select([*cnames, "split"])
    )


def train_test_split_continuous_target(
    data: pl.DataFrame | pl.LazyFrame,
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
    if shuffle_data:
        if isinstance(data, pl.LazyFrame):
            df = data.collect().sample(fraction=1.0, seed=random_seed).lazy()
        else:
            df = data.sample(fraction=1.0, seed=random_seed)
    else:
        df = data.select(pl.all())

    row_count = df.select(pl.len())
    if isinstance(row_count, pl.LazyFrame):
        row_count = row_count.collect()

    row_count = row_count.item()

    # Add a row_number partitioned by the target and use it to compute the splits
    # We can then use the row number to select the desired number of samples for the
    # test set with the correct proportion of the target
    return df.with_row_index(offset=1).with_columns(
        split=pl.when(pl.col("index") <= int(test_size * row_count))
        .then(pl.lit("test"))
        .otherwise(pl.lit("train"))
    )


def univariate_feature_selection_classification(
    data: pl.DataFrame | pl.LazyFrame,
    target_column: str,
    threshold: float = 0.5,
) -> pl.DataFrame | pl.LazyFrame:
    """Perform a preselection of features based on the ROC AUC score of
    a univariate model.

    As the AUC just calculates the quality of a ranking, all monotonous transformations
    of a given ranking (i.e. transformations that do not alter the ranking itself) will
    lead to the same AUC. Hence, training a logistic regression model on a single
    categorical variable (incl. a discretized continuous variable) will produce exactly
    the same ranking as using the target encoded values as model scores. In fact, both
    will produce the exact same output: a ranking of the categories on the training set.
    Therefore, no model is trained here as the target encoded data is/must be used as
    inputs for this function. These will be used as predicted scores to compute the
    ROC AUC with against the target.

    Parameters
    ----------
    data: pl.DataFrame | pl.LazyFrame,
        Input data
    target_column: str
        Name of the target column.
    threshold : float, optional
        Threshold on min. AUC to select the features

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with features dropped that were not selected.
    """
    auc = pl.concat(
        [
            data.select(
                cname=pl.lit(cname),
                # part of polars-ds package:
                roc_auc=pl.col(target_column).metric.roc_auc(pl.col(cname)),
            )
            for cname in data.columns
            if cname != target_column
        ]
    )

    if isinstance(auc, pl.LazyFrame):
        auc = auc.collect()

    logger.info("ROC AUC preselection:")
    logger.info(auc)

    dropped_features = auc.filter(pl.col("roc_auc") <= threshold).select("cname")

    dropped_features = dropped_features.to_dict(as_series=False)["cname"]

    return data.select(
        [cname for cname in data.columns if cname not in dropped_features]
    )


def univariate_feature_selection_regression(
    data: pl.DataFrame | pl.LazyFrame,
    target_column: str,
    threshold: float = 5,
) -> pl.DataFrame | pl.LazyFrame:
    """Perform a preselection of features based on the RMSE score of a univariate model.

    Parameters
    ----------
    data: pl.DataFrame | pl.LazyFrame,
        Input data
    target_column: str
        Name of the target column.
    threshold : float, optional
        Threshold on max. RMSE to select the features

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        DataFrame with features dropped that were not selected.
    """
    num_rows = data.select(pl.len()).lazy().collect().item()

    rmse = pl.concat(
        [
            data.select(
                cname=pl.lit(cname),
                # part of polars-ds package:
                rmse=((pl.col(target_column) - pl.col(cname)) ** 2 / num_rows)
                .sum()
                .sqrt(),
            )
            for cname in data.columns
            if cname != target_column
        ]
    )

    if isinstance(rmse, pl.LazyFrame):
        rmse = rmse.collect()

    logger.info("RMSE preselection:")
    logger.info(rmse)

    dropped_features = rmse.filter(pl.col("rmse") >= threshold).select("cname")
    dropped_features = dropped_features.to_dict(as_series=False)["cname"]

    return data.select(
        [cname for cname in data.columns if cname not in dropped_features]
    )
