# standard lib imports
import numbers
import operator
from itertools import pairwise

# third party imports
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class KBinsDiscretizer(BaseEstimator, TransformerMixin):
    """Bin continuous data into intervals of predefined size.

    It provides a way to partition continuous data into discrete values, i.e. transform
    continuous data into nominal data. This can make a linear model more expressive as
    it introduces nonlinearity to the model, while maintaining the interpretability
    of the model afterwards.

    This module is a rework of
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_discretization.py,
    though it is purely written in polars instead of numpy because it is more intuitive.
    It also includes some custom modifications to align it with our methodology.
    For example, we fill missing values with a category ``Missing``. Furthermore, we
    have an option to adapt the number of bins requested based on the number of Null/Nan
    values in a column.

    Attributes
    ----------
    auto_adapt_bins : bool
        Reduces the number of bins (starting from n_bins) as a function of
        the number of Null/Nan values.
    column_names: list
        Names of the columns of the DataFrame to discretize
    label_format : str
        Format string to display the bin labels e.g. ``min - max``, ``(min, max]``, ...
        Defaults to None which then uses ``(min, max]``.
    left_closed : bool
        Set the intervals to be left-closed instead of right-closed.
    n_bins : int
        Number of bins to produce. Raises ValueError if ``n_bins < 2``. A warning
        is issued when a variable can only produce a lower number of bins than
        asked for.
    starting_precision : int
        Initial precision for the bin edges to start from,
        can also be negative. Given a list of bin edges, the class will
        automatically choose the minimal precision required to have proper bins
        e.g. ``[5.5555, 5.5744, ...]`` will be rounded to
        ``[5.56, 5.57, ...]``. In case of a negative number, an attempt will be
        made to round up the numbers of the bin edges e.g. ``5.55 -> 10``,
        ``146 -> 100``, ...
    strategy : str
        Binning strategy. Currently only `uniform` and `quantile`
        e.g. equifrequency is supported.
    """

    valid_strategies = ("uniform", "quantile")

    def __init__(
        self,
        column_names: list,
        n_bins: int = 10,
        strategy: str = "quantile",
        left_closed: bool = False,
        auto_adapt_bins: bool = False,
        starting_precision: int = 1,
        label_format: str | None = None,
    ):
        """Constructor for KBinsDiscretizer"""
        self.column_names = column_names
        # validate number of bins
        self._validate_n_bins(n_bins)

        self.n_bins = n_bins
        self.strategy = strategy.lower()

        if self.strategy not in self.valid_strategies:
            raise ValueError(
                f"{KBinsDiscretizer.__name__}: valid options for 'strategy' are {self.valid_strategies}. "
                f"Got strategy={self.strategy!r} instead."
            )

        self.left_closed = left_closed
        self.auto_adapt_bins = auto_adapt_bins
        self.starting_precision = starting_precision
        self.label_format = label_format

        # dict to store fitted output in
        self.bin_edges_by_column_ = {}
        self.bin_labels_by_column_ = {}

    def fit(self, X: pl.LazyFrame | pl.DataFrame, y=None):
        """Fits the estimator

        Parameters
        ----------
        X : pl.LazyFrame | pl.DataFrame
            Data to be discretized
        y: placeholder for compatibility with scikit-learn's TransformerMixin
        """
        cname_list = [
            cname for cname in self.column_names if cname in X.collect_schema().names()
        ]

        n_bins_by_column = {cname: self.n_bins for cname in cname_list}
        if self.auto_adapt_bins:
            # compute percentage of Null/Nan values and use that to adapt bin size
            # per column
            res = (
                X.lazy()
                .select((pl.all().is_null().sum() + pl.all().is_nan().sum()) / pl.len())
                .collect()
            )

            missing_pct_by_col = {
                k: v[0]
                for k, v in res.to_dict(as_series=False).items()
                if k in cname_list
            }
            n_bins_by_column = {
                cname: int(max(round((1 - missing_pct_by_col[cname]) * self.n_bins), 2))
                for cname in cname_list
            }

        if self.strategy == "uniform":
            self._fit_with_uniform_strategy(X, n_bins_by_column)
        else:
            # already tested separately
            self._fit_with_quantile_strategy(X, n_bins_by_column)  # pragma: no cover

    def transform(self, X: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        """Discretizes the data in the given list of columns by mapping each
        number to the appropriate bin computed by the fit method

        Parameters
        ----------
        X : data: pl.LazyFrame | pl.DataFrame
            Data to be discretized

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with discretized variables
        """
        if len(self.bin_edges_by_column_) == 0:
            msg = (
                "{} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )

            raise NotFittedError(msg.format(self.__class__.__name__))

        def cut(cname: str, edges: list, labels: list) -> pl.Expr:
            """Custom implementation for the experimental pl.cut expression"""
            labels = [pl.lit(x, pl.Categorical) for x in labels]

            operation = operator.le
            if self.left_closed:  # pragma: no cover
                operation = operator.lt

            expr = pl.when(pl.col(cname).is_null() | pl.col(cname).is_nan()).then(
                pl.lit("Missing", pl.Categorical)
            )
            for edge, label in zip(edges, labels[:-1], strict=False):
                expr = expr.when(operation(pl.col(cname), edge)).then(label)
            expr = expr.otherwise(labels[-1])

            return expr

        X = X.with_columns(
            cut(
                cname=cname,
                edges=self.bin_edges_by_column_[cname],
                labels=self.bin_labels_by_column_[cname],
            ).alias(cname)
            for cname in self.bin_edges_by_column_
            if cname in X.collect_schema().names()
        )

        return X

    def fit_transform(
        self, X: pl.LazyFrame | pl.DataFrame, y=None
    ) -> pl.LazyFrame | pl.DataFrame:
        """Fits to data, then transform it

        Parameters
        ----------
        X : pl.LazyFrame | pl.DataFrame
            Data to be discretized
        y: placeholder for compatibility with scikit-learn's TransformerMixin

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with discretized variables
        """
        self.fit(X)
        return self.transform(X)

    def _fit_with_uniform_strategy(
        self, data: pl.LazyFrame | pl.DataFrame, n_bins_by_column: dict
    ):
        """Fits the estimator using the uniform strategy.

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to be discretized
        n_bins_by_column : dict
            mapping of column name to number of bins for that particular column

        """
        min_max = data.select(
            pl.all().min().name.suffix("_min"), pl.all().max().name.suffix("_max")
        )

        if isinstance(min_max, pl.LazyFrame):
            min_max = min_max.collect()

        min_max_dict = min_max.to_dict(as_series=False)

        for cname in n_bins_by_column:
            bin_edges = list(
                np.linspace(
                    min_max_dict[f"{cname}_min"][0],
                    min_max_dict[f"{cname}_max"][0],
                    n_bins_by_column[cname] + 1,
                )
            )

            precision = self._compute_minimal_precision_of_bin_edges(bin_edges)

            # drop first (resp. last) element from bin_edges as we will always use
            # smaller than (resp. bigger than) the second (resp. second to last) element
            self.bin_edges_by_column_[cname] = [
                round(edge, precision) for edge in bin_edges
            ][1:-1]

            # create bin labels
            self.bin_labels_by_column_[cname] = self._create_bin_labels_from_edges(
                self.bin_edges_by_column_[cname]
            )

    def _fit_with_quantile_strategy(
        self, data: pl.LazyFrame | pl.DataFrame, n_bins_by_column: dict
    ):
        """Fits the estimator using the quantile strategy.

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to be discretized
        n_bins_by_column : dict
            mapping of column name to number of bins for that particular column

        """
        # In the list of quantiles to compute, we exclude edges as these are the min/max
        # of the column which we won't use since we will handle edges differently
        res = pl.concat(
            [
                data.select(
                    cname=pl.lit(cname),
                    bin_edges=pl.concat_list(
                        [
                            pl.col(cname).quantile(q, interpolation="linear")
                            for q in np.linspace(0, 1, n_bins_by_column[cname] + 1)[
                                1:-1
                            ]
                        ]
                    ),
                )
                for cname in n_bins_by_column
            ],
        )

        if isinstance(res, pl.LazyFrame):
            res = res.collect()

        # list of dicts!
        bin_edges_dict = res.to_dict(as_series=False)

        for cname, bin_edges_raw in zip(
            bin_edges_dict["cname"], bin_edges_dict["bin_edges"], strict=False
        ):
            bin_edges = sorted(list(set(bin_edges_raw)))

            precision = self._compute_minimal_precision_of_bin_edges(bin_edges)

            self.bin_edges_by_column_[cname] = [
                round(edge, precision) for edge in bin_edges
            ]

            # create bin labels
            self.bin_labels_by_column_[cname] = self._create_bin_labels_from_edges(
                self.bin_edges_by_column_[cname]
            )

    def _validate_n_bins(self, n_bins: int):
        """Check if ``n_bins`` is of the proper type and if it is bigger
        than two

        Parameters
        ----------
        n_bins : int
            Number of bins KBinsDiscretizer has to produce for each variable

        Raises
        ------
        ValueError
            in case ``n_bins`` is not an integer or if ``n_bins < 2``
        """
        if not isinstance(n_bins, numbers.Integral):
            raise ValueError(
                f"{KBinsDiscretizer.__name__} received an invalid n_bins type. "
                f"Received {type(n_bins).__name__}, expected int."
            )
        if n_bins < 2:
            raise ValueError(
                f"{KBinsDiscretizer.__name__} received an invalid number "
                f"of bins. Received {n_bins}, expected at least 2."
            )

    def _compute_minimal_precision_of_bin_edges(self, bin_edges: list) -> int:
        """Compute the minimal precision of a list of bin_edges so that we end
        up with a strictly ascending sequence of different numbers even when rounded.
        The starting_precision attribute will be used as the initial precisio1.
        In case of a negative starting_precision, the bin edges will be rounded
        to the nearest 10, 100, ... (e.g. 5.55 -> 10, 246 -> 200, ...)

        Parameters
        ----------
        bin_edges : list
            The bin edges for binning a continuous variable

        Returns
        -------
        int
            minimal precision for the bin edges
        """
        precision = self.starting_precision
        while True:
            cont = False
            for a, b in pairwise(bin_edges):
                if a != b and round(a, precision) == round(b, precision):
                    # precision is not high enough, so increase
                    precision += 1
                    cont = True  # set cont to True to keep looping
                    break  # break out of the for loop
            if not cont:
                # if minimal precision was found,
                # return to break out of while loop
                return precision

    def _create_bin_labels_from_edges(self, bin_edges: list) -> list:
        """Given a list of bin edges, create a list containing the label for each bin.

        This label is a string with a specific format.

        Parameters
        ----------
        bin_edges : List
            list of bin edges

        Returns
        -------
        list
            list of (formatted) bin labels
        """
        label_format = self.label_format
        if label_format is None:
            # Format first and last bin with -inf resp. inf.
            # and properly set label format!
            if self.left_closed:
                first_label = f"(-inf, {bin_edges[0]})"
                last_label = f"[{bin_edges[-1]}, inf)"

                label_format = "[{}, {})"
            else:
                first_label = f"(-inf, {bin_edges[0]}]"
                last_label = f"({bin_edges[-1]}, inf)"

                label_format = "({}, {}]"
        else:  # noqa: PLR5501
            # Format first and last bin as < x and > y resp.
            if self.left_closed:
                first_label = f"< {bin_edges[0]}"
                last_label = f">= {bin_edges[-1]}"
            else:
                first_label = f"<= {bin_edges[0]}"
                last_label = f"> {bin_edges[-1]}"

        bin_labels = [
            label_format.format(interval[0], interval[1])
            for interval in zip(bin_edges, bin_edges[1:], strict=False)  # noqa: RUF007
        ]

        return [first_label, *bin_labels, last_label]
