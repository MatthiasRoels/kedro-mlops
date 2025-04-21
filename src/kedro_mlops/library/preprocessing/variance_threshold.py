import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class VarianceThreshold(BaseEstimator, TransformerMixin):
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

    def fit(self, X: pl.DataFrame | pl.LazyFrame, y=None):
        """Learn empircal variances of the data

        To also include non-numeric data, we will first transform it into numeric data
        using ordinal encoding.

        Parameters
        ----------
        X : pl.LazyFrame | pl.DataFrame
            Data from which to compute variances
        y: placeholder for compatibility with scikit-learn's TransformerMixin
        """
        variances = X.select(
            cs.numeric(),
            cs.exclude(cs.numeric()).rank(method="dense"),
        ).select(pl.all().var())

        if isinstance(variances, pl.LazyFrame):
            variances = variances.collect()

        # do transpose here as a LazyFrame doesn't have such a method
        raw = variances.transpose(include_header=True).to_dict(as_series=False)

        self.columns_to_drop_ = [
            cname
            for cname, var in zip(raw["column"], raw["column_0"], strict=False)
            if var <= self.threshold
        ]

    def transform(self, X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Reduce data to only include selected features

        Parameters
        ----------
        X : pl.LazyFrame | pl.DataFrame
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

        return X.select(
            cname
            for cname in X.collect_schema().names()
            if cname not in self.columns_to_drop_
        )

    def fit_transform(
        self, X: pl.DataFrame | pl.LazyFrame, y=None
    ) -> pl.DataFrame | pl.LazyFrame:
        """Fits to data, then transform it

        Parameters
        ----------
        X : pl.LazyFrame | pl.DataFrame
            Data to transform
        y: placeholder for compatibility with scikit-learn's TransformerMixin

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with only selected features
        """
        self.fit(X)
        return self.transform(X)
