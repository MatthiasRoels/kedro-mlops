# standard lib imports
import logging
import numbers
import operator
from itertools import pairwise

# third party imports
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

log = logging.getLogger(__name__)


class KBinsDiscretizer(BaseEstimator):
    """Bin continuous data into intervals of predefined size.

    It provides a way to partition continuous data into discrete values, i.e. transform
    continuous data into nominal data. This can make a linear model more expressive as
    it introduces nonlinearity to the model, while maintaining the interpretability
    of the model afterwards.

    This module is a rework of
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_discretization.py,
    though it is purely written in polars instead of numpy because it is more intuitive.
    It also includes some custom modifications to align it with our methodology.
    See the README of the GitHub repository for more background information.

    Attributes
    ----------
    auto_adapt_bins : bool
        Reduces the number of bins (starting from n_bins) as a function of
        the number of Null/Nan values.
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
        n_bins: int = 10,
        strategy: str = "quantile",
        left_closed: bool = False,
        auto_adapt_bins: bool = False,
        starting_precision: int = 0,
        label_format: str | None = None,
    ):

        # validate number of bins
        self._validate_n_bins(n_bins)

        self.n_bins = n_bins
        self.strategy = strategy.lower()

        if self.strategy not in self.valid_strategies:
            raise ValueError(
                "{}: valid options for 'strategy' are {}. "
                "Got strategy={!r} instead.".format(
                    KBinsDiscretizer.__name__, self.valid_strategies, self.strategy
                )
            )

        self.left_closed = left_closed
        self.auto_adapt_bins = auto_adapt_bins
        self._starting_precision = starting_precision
        self._label_format = label_format

        # dict to store fitted output in
        self.bin_edges_by_column_ = {}
        self.bin_labels_by_column_ = {}

    def fit(self, data: pl.LazyFrame | pl.DataFrame, column_names: list):
        """Fits the estimator

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to be discretized
        column_names : list
            Names of the columns of the DataFrame to discretize
        """
        cname_list = [cname for cname in column_names if cname in data.columns]

        n_bins_by_column = {cname: self.n_bins for cname in cname_list}
        if self.auto_adapt_bins:
            # compute percentage of Null/Nan values and use that to adapt bin size
            # per column
            res = data.select((pl.all().is_null() + pl.all().is_nan()).sum() / pl.len())
            if isinstance(res, pl.LazyFrame):
                res = res.collect()
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
            self._fit_with_uniform_strategy(data, n_bins_by_column)
        else:
            self._fit_with_quantile_strategy(data, n_bins_by_column)

    def _fit_with_uniform_strategy(
        self, data: pl.LazyFrame | pl.DataFrame, n_bins_by_column: dict
    ):
        min_max = data.select(
            pl.all().min().name.suffix("_min"), pl.all().max().name.suffix("_max")
        )

        if isinstance(min_max, pl.LazyFrame):
            min_max = min_max.collect()

        min_max_dict = min_max.to_dict(as_series=False)

        for cname in n_bins_by_column:
            bin_edges = list(
                np.linspace(
                    min_max_dict[f"{cname}_min"],
                    min_max_dict[f"{cname}_max"],
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
            self._bin_labels_by_column[cname] = self._create_bin_labels_from_edges(
                bin_edges
            )

    def _fit_with_quantile_strategy(
        self, data: pl.LazyFrame | pl.DataFrame, n_bins_by_column: dict
    ):
        pass

    def transform(
        self, data: pl.LazyFrame | pl.DataFrame
    ) -> pl.LazyFrame | pl.DataFrame:
        """Discretizes the data in the given list of columns by mapping each
        number to the appropriate bin computed by the fit method

        Parameters
        ----------
        data : data: pl.LazyFrame | pl.DataFrame
            Data to be discretized

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with discretized variables
        """
        if len(self.bins_by_column_) == 0:
            msg = (
                "{} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )

            raise NotFittedError(msg.format(self.__class__.__name__))

        def cut(cname: str, edges: list, labels: list) -> pl.Expr:

            labels = [pl.lit(x, pl.Categorical) for x in labels]

            operator = operator.le
            if self.left_closed:
                operator = operator.lt

            expr = pl.when(pl.col(cname).is_null() | pl.col(cname).is_nan()).then(
                pl.lit("Missing", pl.Categorical)
            )
            for edge, label in zip(edges, labels[:-1]):
                expr = expr.when(operator(pl.col(cname), edge)).then(label)
            expr = expr.otherwise(labels[-1])

            return expr

        data = data.with_columns(
            cut(
                cname=cname,
                edges=self.bin_edges_by_column_[cname],
                labels=self.bin_labels_by_column_[cname],
            ).alias(cname)
            for cname in self.bin_edges_by_column_
            if cname in data.columns
        )

        return data

    def fit_transform(
        self, data: pl.LazyFrame | pl.DataFrame, column_names: list
    ) -> pl.LazyFrame | pl.DataFrame:
        """Fits to data, then transform it

        Parameters
        ----------
        data : pl.LazyFrame | pl.DataFrame
            Data to be discretized
        column_names : list
            Names of the columns of the DataFrame to discretize

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            data with discretized variables
        """
        self.fit(data, column_names)
        return self.transform(data, column_names)

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
        The starting_precision attribute will be used as the initial precision.
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
        precision = self._starting_precision
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

    def _compute_bins_from_edges(self, bin_edges: list) -> list[tuple]:
        """Given a list of bin edges, compute the minimal precision for which
        we can make meaningful bins and make those bins

        Parameters
        ----------
        bin_edges : list
            The bin edges for binning a continuous variable

        Returns
        -------
        List[tuple]
            A (sorted) list of bins as tuples
        """
        # compute the minimal precision of the bin_edges
        # this can be a negative number, which then
        # rounds numbers to the nearest 10, 100, ...
        precision = self._compute_minimal_precision_of_bin_edges(bin_edges)

        bins = []
        for a, b in pairwise(bin_edges):
            fmt_a = round(a, precision)
            fmt_b = round(b, precision)

            bins.append((fmt_a, fmt_b))

        return bins

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
        label_format = self._label_format if self._label_format else "({}, {}]"
        bin_labels = [
            label_format.format(interval[0], interval[1])
            for interval in zip(bin_edges, bin_edges[1:])
        ]

        # Format first and last bin with -inf resp. inf.
        if self._label_format is None:
            if self.left_closed:
                first_label = f"(-inf, {bin_edges[0]}]"
                last_label = f"({bin_edges[-1]}, inf)"
            else:
                first_label = f"(-inf, {bin_edges[0]})"
                last_label = f"[{bin_edges[-1]}, inf]"
        # Format first and last bin as < x and > y resp.
        else:  # noqa: PLR5501
            if self.left_closed:
                first_label = f"< {bin_edges[0]}"
                last_label = f">= {bin_edges[-1]}"
            else:
                first_label = f"<= {bin_edges[0]}"
                last_label = f"> {bin_edges[-1]}"

        return [first_label, *bin_labels, last_label]
