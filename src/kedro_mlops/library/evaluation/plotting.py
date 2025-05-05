import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats


def plot_roc_curve(fpr, tpr, auc: float, dim: tuple | None = None) -> Figure:
    """Plot ROC curve of the model"""
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=dim)

        label = f"ROC curve (area = {auc:.3f})"
        ax.plot(fpr, tpr, color="royalblue", linewidth=3, label=label)

        ax.plot(
            [0, 1],
            [0, 1],
            color="red",
            linewidth=3,
            linestyle="--",
            label="random selection",
        )
        ax.set_xlabel("False positive rate", fontsize=15)
        ax.set_ylabel("True positive rate", fontsize=15)
        ax.legend(loc="lower right")
        ax.set_title("ROC curve", fontsize=20)

        ax.set_ylim([0, 1])

    plt.close(fig)
    return fig


def plot_pr_curve(
    precision, recall, average: float, dim: tuple | None = None
) -> Figure:
    """Plot Precision Recall curve of the model"""
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=dim)

        label = f"PR curve (Average Precision = {average:.3f})"
        ax.plot(recall, precision, color="royalblue", linewidth=3, label=label)

        ax.set_xlabel("Recall", fontsize=15)
        ax.set_ylabel("Precision", fontsize=15)
        ax.legend(loc="lower right")
        ax.set_title("PR curve", fontsize=20)

        ax.set_ylim([0, 1])

    plt.close(fig)
    return fig


def plot_confusion_matrix(
    confusion_matrix, dim: tuple | None = None, labels: list | None = None
) -> Figure:
    """Plot the confusion matrix"""
    if dim is None:
        dim = (12, 8)
    if labels is None:  # pragma: no cover
        # Default labels for binary classification
        labels = ["0", "1"]

    fig, ax = plt.subplots(figsize=dim)
    ax = sns.heatmap(
        confusion_matrix,
        annot=confusion_matrix.astype(str),
        fmt="s",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Confusion matrix", fontsize=20)
    plt.ylabel("True labels", fontsize=15)
    plt.xlabel("Predicted labels", fontsize=15)
    plt.close(fig)
    return fig


def plot_correlation_matrix(df_corr: pd.DataFrame, dim: tuple | None = None):
    """Plot correlation matrix of the predictors"""
    if dim is None:
        dim = (12, 8)

    fig, ax = plt.subplots(figsize=dim)
    ax = sns.heatmap(df_corr, cmap="Blues")
    ax.set_title("Correlation matrix", fontsize=20)

    plt.close(fig)
    return fig


def plot_predictions_vs_actuals(
    y_true: np.array, y_pred: np.array, dim: tuple | None = None
) -> Figure:
    """Plot the predictions vs actuals"""
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=dim)

        x = np.arange(1, len(y_true) + 1)

        ax.plot(x, y_true, ls="--", label="actuals", color="red", linewidth=3)
        ax.plot(x, y_pred, label="predictions", color="royalblue", linewidth=3)

        ax.set_xlabel("Index", fontsize=15)
        ax.set_ylabel("Value", fontsize=15)
        ax.legend(loc="best")
        ax.set_title("Predictions vs. Actuals", fontsize=20)

    plt.close(fig)
    return fig


def plot_qq(y_residual: np.array, dim: tuple | None = None) -> Figure:
    """Q-Q plot of the residuals of the model."""
    if dim is None:
        dim = (12, 8)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=dim)

        stats.probplot(y_residual, plot=ax)
        ax.set_title("Q-Q plot", fontsize=20)

    plt.close(fig)
    return fig


def plot_feature_incidence_graphs(  # noqa: PLR0915
    fig_table: pd.DataFrame,
    variable: str,
    model_type: str,
    column_order: list | None = None,
    dim: tuple | None = None,
):  # pragma: no cover
    """Plots a Feature Insights Graph (FIG), a graph in which the mean
    target value is plotted for a number of bins constructed from a predictor
    variable. When the target is a binary classification target,
    the plotted mean target value is a true incidence rate.

    Bins are ordered in descending order of mean target value
    unless specified otherwise with the `column_order` list.

    Parameters
    ----------
    fig_table: pd.DataFrame
        Dataframe with cleaned, binned, partitioned and prepared data,
        as created by `generate_fig_tables`.
    variable: str
        Name of the predictor variable for which the FIG will be plotted.
    model_type: str
        Type of model (either "classification" or "regression").
    column_order: list, default=None
        Explicit order of the value bins of the predictor variable to be used
        on the FIG.
    dim: tuple, default=(12, 8)
        Optional tuple to configure the width and length of the plot.
    """
    if model_type not in ["classification", "regression"]:
        raise ValueError(
            "An unexpected value was set for the model_type "
            "parameter. Expected 'classification' or "
            "'regression'."
        )

    if dim is None:
        dim = (12, 8)

    if column_order is not None:
        if not set(fig_table["label"]) == set(column_order):
            raise ValueError(
                "The column_order and pig_tables parameters do not contain "
                "the same set of variables."
            )

        fig_table["label"] = fig_table["label"].astype("category")
        fig_table["label"].cat.reorder_categories(column_order, inplace=True)

        df_plot = fig_table.sort_values(by=["label"], ascending=True).reset_index()
    else:
        df_plot = fig_table.sort_values(
            by=["avg_target"], ascending=False
        ).reset_index()

    with plt.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=dim)

        # --------------------------
        # Left axis - average target
        # --------------------------
        ax.plot(
            df_plot["label"],
            df_plot["avg_target"],
            color="#00ccff",
            marker=".",
            markersize=20,
            linewidth=3,
            label="incidence rate per bin"
            if model_type == "classification"
            else "mean target value per bin",
            zorder=10,
        )

        ax.plot(
            df_plot["label"],
            df_plot["global_avg_target"],
            color="#022252",
            linestyle="--",
            linewidth=4,
            label="average incidence rate"
            if model_type == "classification"
            else "global mean target value",
            zorder=10,
        )

        # Dummy line to have label on second axis from first
        ax.plot(np.nan, "#939598", linewidth=6, label="bin size")

        # Set labels & ticks
        ax.set_ylabel(
            "Incidence" if model_type == "classification" else "Mean target value",
            fontsize=16,
        )
        ax.set_xlabel("Bins", fontsize=15)
        ax.xaxis.set_tick_params(labelsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.yaxis.set_tick_params(labelsize=14)

        if model_type == "classification":
            # Mean target values are between 0 and 1 (target incidence rate),
            # so format them as percentages
            ax.set_yticks(np.arange(0, max(df_plot["avg_target"]) + 0.05, 0.05))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
        elif model_type == "regression":
            # If the difference between the highest avg. target of all bins
            # versus the global avg. target AND the difference between the
            # lowest avg. target versus the global avg. target are both smaller
            # than 25% of the global avg. target itself, we increase the
            # y-axis range, to avoid that the minor avg. target differences are
            # spread out over the configured figure height, suggesting
            # incorrectly that there are big differences in avg. target across
            # the bins and versus the global avg. target.
            # (Motivation for the AND above: if on one end there IS enough
            # difference, the effect that we discuss here does not occur.)
            global_avg_target = max(
                df_plot["global_avg_target"]
            )  # series of same number, for every bin.
            if (
                np.abs(max(df_plot["avg_target"]) - global_avg_target)
                / global_avg_target
                < 0.25
            ) and (
                np.abs(min(df_plot["avg_target"]) - global_avg_target)
                / global_avg_target
                < 0.25
            ):
                ax.set_ylim(global_avg_target * 0.75, global_avg_target * 1.25)

        # Remove ticks but keep the labels
        ax.tick_params(axis="both", which="both", length=0)
        ax.tick_params(axis="y", colors="#00ccff")
        ax.yaxis.label.set_color("#00ccff")

        # -----------------
        # Right Axis - bins
        # -----------------
        ax2 = ax.twinx()

        ax2.bar(
            df_plot["label"],
            df_plot["pop_size"],
            align="center",
            color="#939598",
            zorder=1,
        )

        # Set labels & ticks
        ax2.set_xlabel("Bins", fontsize=15)
        ax2.xaxis.set_tick_params(rotation=45, labelsize=14)

        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax2.set_ylabel("Population size", fontsize=15)
        ax2.tick_params(axis="y", colors="#939598")
        ax2.yaxis.label.set_color("#939598")

        # Despine & prettify
        sns.despine(ax=ax, right=True, left=True)
        sns.despine(ax=ax2, left=True, right=False)
        ax2.spines["right"].set_color("white")

        ax2.grid(False)

        # Title & legend
        if model_type == "classification":
            title = "Incidence plot"
        else:
            title = "Mean target plot"
        fig.suptitle(title, fontsize=20)
        plt.title(variable, fontsize=17)
        ax.legend(
            frameon=False,
            bbox_to_anchor=(0.0, 1.01, 1.0, 0.102),
            loc=3,
            ncol=1,
            mode="expand",
            borderaxespad=0.0,
            prop={"size": 14},
        )

        # Set order of layers
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        plt.tight_layout()
        plt.margins(0.01)

    plt.close(fig)
    return fig
