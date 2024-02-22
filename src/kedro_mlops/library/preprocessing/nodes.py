import polars as pl


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
        str(k): v for k, v in zip(row_counts_raw["target"], row_counts_raw["len"])
    }

    # Add a row_number partitioned by the target and use it to compute the splits
    # We can then use the row number to select the desired number of samples for the
    # test set with the correct proportion of the target
    return df.with_columns(
        row_number=pl.col(example_col).rank(method="ordinal").over(target)
    ).with_columns(
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
    ).select([*cnames, "split"])
