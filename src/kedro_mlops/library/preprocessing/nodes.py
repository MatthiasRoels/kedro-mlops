import polars as pl
from sklearn.base import BaseEstimator

from kedro_mlops.library.utils import materialize_data

from .kbins_discretizer import KBinsDiscretizer
from .target_encoder import TargetEncoder
from .utils import univariate_feature_selection_classification
from .variance_threshold import VarianceThreshold


def apply_variance_threshold(
    data: pl.DataFrame | pl.LazyFrame,
    target_column: str | None = None,
    threshold: float = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Wrapper function around VarianceThreshold

    Parameters
    ----------
    data : pl.LazyFrame | pl.DataFrame
        Data on which a feature pre-selection is performed
    target_column: str
        Column name of the target.
    threshold: float
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Returns
    -------
    pl.LazyFrame | pl.DataFrame
        data with only selected features
    """
    variance_threshold = VarianceThreshold(threshold)
    variance_threshold.fit(
        data.filter(pl.col("split") == "train").select(
            pl.exclude(["split", target_column])
        )
    )
    return variance_threshold.transform(data)


def fit_discretizer(
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
    discretizer = KBinsDiscretizer(
        **{**discretizer_config, "column_names": column_names}
    )
    discretizer.fit(data.filter(pl.col("split") == "train"))

    return discretizer


def fit_encoder(
    data: pl.DataFrame | pl.LazyFrame,
    target_column: str,
    encoder_config: dict,
    pk_col: str | list | None = None,
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
    excluded = ["split", target_column]
    if isinstance(pk_col, str):
        excluded.append(pk_col)
    elif isinstance(pk_col, list):
        excluded = [*excluded, *pk_col]

    column_names = [
        cname for cname in data.collect_schema().names() if cname not in excluded
    ]
    encoder = TargetEncoder(**{**encoder_config, "column_names": column_names})

    encoder.fit(
        X=data.filter(pl.col("split") == "train").select(column_names),
        y=data.filter(pl.col("split") == "train").select(target_column),
    )

    return encoder


def transform_data(
    data: pl.DataFrame | pl.LazyFrame,
    fitted_estimator: BaseEstimator,
) -> pl.DataFrame | pl.LazyFrame:
    """Wrapper around transform methods of estimators"""
    return fitted_estimator.transform(data)


def prepare_train_data(
    data: pl.DataFrame | pl.LazyFrame,
    target_column: str,
    threshold: float,
) -> pl.DataFrame:
    """Wrapper around univariate_feature_selection_classification utils function

    This is the final preparation step to generate a ready-to-use training dataset.

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
    pl.DataFrame
        Pepared dataset for training
    """
    train_data = data.filter(pl.col("split") == "train").select(
        [
            cname
            for cname in data.columns
            if cname.endswith("_enc") or cname == target_column
        ]
    )

    prepared_train_data = univariate_feature_selection_classification(
        train_data, target_column, threshold
    )

    return materialize_data(prepared_train_data)
