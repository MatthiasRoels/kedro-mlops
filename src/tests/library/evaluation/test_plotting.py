import os

import numpy as np
import pandas as pd
import pytest

from src.kedro_mlops.library.evaluation.plotting import (
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


@pytest.fixture
def tmp_file_path(tmp_path):
    """Fixture to create a temporary file path."""
    return os.path.join(tmp_path, "test_plot.png")


def test_plot_roc_curve(tmp_file_path):
    fpr = [0.0, 0.1, 0.2, 1.0]
    tpr = [0.0, 0.4, 0.8, 1.0]
    auc = 0.85

    plot_roc_curve(fpr, tpr, auc, tmp_file_path)

    assert os.path.exists(tmp_file_path)


def test_plot_pr_curve(tmp_file_path):
    precision = [1.0, 0.8, 0.6, 0.0]
    recall = [0.0, 0.4, 0.8, 1.0]
    average = 0.75

    plot_pr_curve(precision, recall, average, tmp_file_path)

    assert os.path.exists(tmp_file_path)


def test_plot_confusion_matrix(tmp_file_path):
    confusion_matrix = np.array([[50, 10], [5, 35]])
    labels = ["Class 0", "Class 1"]

    plot_confusion_matrix(confusion_matrix, tmp_file_path, labels=labels)

    assert os.path.exists(tmp_file_path)


def test_plot_correlation_matrix(tmp_file_path):
    df_corr = pd.DataFrame(
        {
            "Feature1": [1.0, 0.8, 0.5],
            "Feature2": [0.8, 1.0, 0.3],
            "Feature3": [0.5, 0.3, 1.0],
        }
    )

    plot_correlation_matrix(df_corr, tmp_file_path)

    assert os.path.exists(tmp_file_path)
