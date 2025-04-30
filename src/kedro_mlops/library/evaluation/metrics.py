import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_classfier_metrics(
    y_true, y_pred, y_pred_b
) -> dict[str, float]:  # pragma: no cover
    return {
        "accuracy": accuracy_score(y_true, y_pred_b),
        "AUC": roc_auc_score(y_true, y_pred),
        "Average Precision score": average_precision_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred_b),
        "recall": recall_score(y_true, y_pred_b),
        "F1": f1_score(y_true, y_pred_b, average=None)[1],
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred_b),
        "lift at 5 percent": np.round(
            compute_lift(y_true=y_true, y_pred=y_pred, lift_at=0.05), 2
        ),
        "lift at 10 percent": np.round(
            compute_lift(y_true=y_true, y_pred=y_pred, lift_at=0.1), 2
        ),
    }


def compute_regressor_metrics(y_true, y_pred) -> dict[str, float]:  # pragma: no cover
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def compute_lift(y_true: np.ndarray, y_pred: np.ndarray, lift_at: float) -> float:
    """Calculates lift given two arrays on specified level.

    Parameters
    ----------
    y_true : np.ndarray
        True binary target data labels.
    y_pred : np.ndarray
        Target scores of the model.
    lift_at : float, optional
        At what top level percentage the lift should be computed.

    Returns
    -------
    float
        Lift of the model.
    """
    if lift_at > 1 or lift_at < 0:
        raise ValueError("lift_at should be between 0 and 1")

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred should not be empty")

    # Make sure it is numpy array
    y_true_ = np.array(y_true)
    y_pred_ = np.array(y_pred)

    # Make sure it has correct shape
    y_true_ = y_true_.reshape(len(y_true_), 1)
    y_pred_ = y_pred_.reshape(len(y_pred_), 1)

    # Merge data together
    y_data = np.hstack([y_true_, y_pred_])

    # Calculate necessary variables
    nrows = len(y_data)
    stop = int(np.floor(nrows * lift_at))
    avg_incidence = np.einsum("ij->j", y_true_) / float(len(y_true_))

    # Sort and filter data
    data_sorted = y_data[y_data[:, 1].argsort()[::-1]][:stop, 0].reshape(stop, 1)

    # Calculate lift (einsum is a very fast way of summing, but needs specific shape)
    inc_in_top_n = np.einsum("ij->j", data_sorted) / float(len(data_sorted))

    lift = np.round(inc_in_top_n / avg_incidence, 2)[0]

    return lift
