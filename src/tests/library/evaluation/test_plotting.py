import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.kedro_mlops.library.evaluation.plotting import (
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def test_plot_roc_curve():
    fpr = [0.0, 0.1, 0.2, 1.0]
    tpr = [0.0, 0.4, 0.8, 1.0]
    auc = 0.85

    fig = plot_roc_curve(fpr, tpr, auc)

    assert isinstance(fig, plt.Figure)


def test_plot_pr_curve():
    precision = [1.0, 0.8, 0.6, 0.0]
    recall = [0.0, 0.4, 0.8, 1.0]
    average = 0.75

    fig = plot_pr_curve(precision, recall, average)

    assert isinstance(fig, plt.Figure)


def test_plot_confusion_matrix():
    confusion_matrix = np.array([[50, 10], [5, 35]])
    labels = ["Class 0", "Class 1"]

    fig = plot_confusion_matrix(confusion_matrix, labels=labels)

    assert isinstance(fig, plt.Figure)


def test_plot_correlation_matrix():
    df_corr = pd.DataFrame(
        {
            "Feature1": [1.0, 0.8, 0.5],
            "Feature2": [0.8, 1.0, 0.3],
            "Feature3": [0.5, 0.3, 1.0],
        }
    )

    fig = plot_correlation_matrix(df_corr)

    assert isinstance(fig, plt.Figure)
