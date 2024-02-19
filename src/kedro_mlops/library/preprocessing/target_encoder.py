import logging

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

log = logging.getLogger(__name__)


class TargetEncoder(BaseEstimator):
    """Target encoding for categorical features

    This was inspired by
    http://contrib.scikit-learn.org/category_encoders/targetencoder.html.

    Replace each value of the categorical feature with the average of the
    target values (in case of a binary target, this is the incidence of the
    group). This encoding scheme is also called Mean encoding.

    Note that, when applying this target encoding, values of the categorical
    feature that have not been seen during fit will be imputed according to the
    configured imputation strategy (replacement with the mean, median, minimum or
    maximum value of the categorical variable).

    The main problem with Target encoding is overfitting; the fact that we are
    encoding the feature based on target classes may lead to data leakage,
    rendering the feature biased. This can be solved using some type of regularization.
    A popular way to handle this is to use cross-validation and compute the means
    in each out-of-fold. However, the approach implemented here makes use of
    additive smoothing (https://en.wikipedia.org/wiki/Additive_smoothing).

    In summary:

    - with a binary classification target, a value of a categorical variable is
    replaced with:

        [count(variable=value) * P(target=1|variable=value) + weight * P(target=1)]
        / [count(variable=value) + weight]

    - with a regression target, a value of a categorical variable is replaced
    with:

        [count(variable=value) * E(target|variable=value) + weight * E(target)]
        / [count(variable=value) + weight]

    Attributes
    ----------
    imputation_strategy : str
        In case there is a particular column which contains new categories,
        the encoding will lead to NULL values which should be imputed.
        Valid strategies then are to replace the NULL values with the global
        mean or the min (resp. max) incidence of the variable.

        Ex: By taking the mean strategy the mean of the known encoded variables
        is computed and the missing encoded values would be imputed with this value.
    weight : float
        Smoothing parameter (non-negative). The higher the value of the
        parameter, the bigger the contribution of the overall mean of targets
        learnt from all training data (prior) and the smaller the contribution
        of the mean target learnt from data with the current categorical value
        (posterior), so the bigger the smoothing (regularization) effect.
        When set to zero, there is no smoothing (e.g. the mean target of the
        current categorical value is used).
    """

    valid_imputation_strategies = ("mean", "min", "max")

    def __init__(self, weight: float = 0.0, imputation_strategy: str = "mean"):

        if weight < 0:
            raise ValueError("The value of weight cannot be smaller than zero.")
        elif imputation_strategy not in self.valid_imputation_strategies:
            raise ValueError(
                "Valid options for 'imputation_strategy' "
                f"are {self.valid_imputation_strategies}. Got "
                f"imputation_strategy={imputation_strategy} instead."
            )

        if weight == 0:
            log.warning(
                "The target encoder's additive smoothing weight is set to 0."
                "This disables smoothing and may make the encoding prone to "
                "overfitting. Increase the weight if needed."
            )

        self.weight = weight
        self.imputation_strategy = imputation_strategy

        self.mapping_ = {}  # placeholder for fitted output
        # placeholder for the global incidence of the data used for fitting
        self.global_mean_ = None

    def fit(
        self, data: pl.LazyFrame | pl.DataFrame, column_names: list, target_column: str
    ):
        """Fit the TargetEncoder to the data.

        Parameters
        ----------
        X : pl.DataFrame | pl.LazyFrame
            train data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be encoded.
        target_column : str
            Column name of the target.
        """
        # compute global mean (target incidence in case of binary target)
        stats = data.select(pl.sum(target_column).alias("sum"), pl.len().alias("count"))

        if isinstance(stats, pl.LazyFrame):
            stats = stats.collect()

        stats = stats.to_dict(as_series=False)

        self.global_mean_ = stats["sum"][0] / stats["count"][0]

        cname_list = [cname for cname in column_names if cname in data.columns]

        res = pl.concat(
            [
                data.group_by(cname)
                .agg(pl.mean(target_column).alias("mean"), pl.count())
                .with_columns(
                    cname=pl.lit(cname),
                    incidence=(
                        pl.col("count") * pl.col("mean")
                        + self.weight * self.global_mean_
                    )
                    / (pl.col("count") + self.weight),
                )
                .select("cname", pl.col(cname).alias("value"), "incidence")
                for cname in cname_list
            ],
            how="diagonal",
        )

        if isinstance(res, pl.LazyFrame):
            res = res.collect()

        # unpack result in a mapping to make it easier to do the transform step
        # efficiently
        for row in res.iter_rows(named=True):
            if row["cname"] in self.mapping_:
                self.mapping_[row["cname"]].update({row["value"]: row["incidence"]})
            else:
                self.mapping_[row["cname"]] = {row["value"]: row["incidence"]}

    def transform(
        self, data: pl.LazyFrame | pl.DataFrame
    ) -> pl.LazyFrame | pl.DataFrame:
        """Replace (e.g. encode) values of each categorical column with a
        new value (reflecting the corresponding average target value,
        optionally smoothed by a regularization weight),
        which was computed when the fit method was called.

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to encode.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The resulting transformed data.

        Raises
        ------
        NotFittedError
            Exception when TargetEncoder was not fitted before calling this
            method.
        """
        if (len(self.mapping_) == 0) or (self.global_mean_ is None):
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        data = data.with_columns(
            pl.col(cname)
            .replace(
                self.mapping_[cname],
                default=self._get_impute_value(list(self.mapping_[cname].values())),
            )
            .alias(self._clean_column_name(cname))
            for cname in self.mapping_
            if cname in data.columns
        )

        return data

    def fit_transform(
        self, data: pl.LazyFrame | pl.DataFrame, column_names: list, target_column: str
    ) -> pl.LazyFrame | pl.DataFrame:
        """Fit the encoder and transform the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be encoded.
        column_names : list
            Columns of data to be encoded.
        target_column : str
            Column name of the target.

        Returns
        -------
        pd.DataFrame
            Data with additional columns, holding the target-encoded variables.
        """
        self.fit(data, column_names, target_column)
        return self.transform(data)

    def _get_impute_value(self, incidences: list) -> float:
        """Impute missing data based on the given strategy.

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to impute.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            Resulting transformed data.
        """
        # In case of categorical data, it could be that new categories will
        # emerge which were not present in the train set, so this will result
        # in missing values, which should be replaced according to the
        # configured imputation strategy:
        if self.imputation_strategy == "mean":
            return self.global_mean_
        elif self.imputation_strategy == "min":
            return min(incidences)
        elif self.imputation_strategy == "max":
            return max(incidences)

    @staticmethod
    def _clean_column_name(column_name: str) -> str:
        """Generate a name for the new column that this target encoder
        generates in the given data, by removing "_bin", "_processed" or
        "_cleaned" from the original categorical column, and adding "_enc".

        Parameters
        ----------
        column_name : str
            Column name to be cleaned.

        Returns
        -------
        str
            Cleaned column name.
        """
        if "_bin" in column_name:
            return column_name.replace("_bin", "") + "_enc"
        elif "_processed" in column_name:
            return column_name.replace("_processed", "") + "_enc"
        elif "_cleaned" in column_name:
            return column_name.replace("_cleaned", "") + "_enc"
        else:
            return column_name + "_enc"
