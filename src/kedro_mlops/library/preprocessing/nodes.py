import polars as pl
from sklearn.base import BaseEstimator

from kedro_mlops.library.utils import materialize_data

from .kbins_discretizer import KBinsDiscretizer
from .target_encoder import TargetEncoder
from .utils import univariate_feature_selection_classification
from .variance_threshold import VarianceThreshold


def apply_variance_threshold(
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
    discretizer = KBinsDiscretizer(**discretizer_config)
    discretizer.fit(data.filter(pl.col("split") == "train"), column_names)

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

    column_names = [cname for cname in data.columns if cname not in excluded]
    encoder = TargetEncoder(**encoder_config)
    encoder.fit(data.filter(pl.col("split") == "train"), column_names, target_column)

    return encoder


def transform_data(
    data: pl.DataFrame | pl.LazyFrame,
    fitted_estimator: BaseEstimator,
) -> pl.DataFrame | pl.LazyFrame:
    """Wrapper around transform methods of estimators"""
    return fitted_estimator.transform(data)


def prepare_train_data(
    data: pl.DataFrame | pl.LazyFrame,
    target: str,
    threshold: float,
) -> pl.DataFrame:
    """Wrapper around univariate_feature_selection_classification utils function

    This is the final preparation step to generate a ready-to-use training dataset.

    Parameters
    ----------
    data: pl.DataFrame | pl.LazyFrame,
        Input data
    target : str
        Name of the target column.
    threshold : float, optional
        Threshold on min. AUC to select the features

    Returns
    -------
    pl.DataFrame
        Pepared dataset for training
    """
    train_data = data.filter(pl.col("split") == "train").select(
        [cname for cname in data.columns if cname.endswith("_enc") or cname == target]
    )

    prepared_train_data = univariate_feature_selection_classification(
        train_data, target, threshold
    )

    return materialize_data(prepared_train_data)
