import polars as pl
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class VarianceThreshold(BaseEstimator):
    """Feature selector that removes all low-variance features.

    It is loosely inspired by
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    but completely rewritten in polars

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning too.

    Attributes
    ----------
    threshold : float, default=0
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.
    """

    def __init__(self, threshold: float = 0):
        self.threshold = threshold

        self.columns_to_drop_ = None

    def fit(self, data: pl.DataFrame | pl.LazyFrame):
        """Learn empircal variances of the data

        To also include non-numeric data, we will first transform it into numeric data
        using ordinal encoding.

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data from which to compute variances
        """
        variances = data.select(
            pl.col(pl.NUMERIC_DTYPES),
            pl.all().exclude(pl.NUMERIC_DTYPES).rank(method="dense"),
        ).select(pl.all().var())

        if isinstance(variances, pl.LazyFrame):
            variances = variances.collect()

        # do transpose here as a LazyFrame doesn't have such a method
        raw = variances.transpose(include_header=True).to_dict(as_series=False)

        self.columns_to_drop_ = [
            cname
            for cname, var in zip(raw["column"], raw["column_0"])
            if var <= self.threshold
        ]

    def transform(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Reduce data to only include selected features

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to transform

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with only selected features
        """
        if self.columns_to_drop_ is None:
            msg = (
                "{} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )

            raise NotFittedError(msg.format(self.__class__.__name__))

        return data.select(
            cname for cname in data.columns if cname not in self.columns_to_drop_
        )

    def fit_transform(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Fits to data, then transform it

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to transform

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with only selected features
        """
        self.fit(data)
        return self.transform(data)
