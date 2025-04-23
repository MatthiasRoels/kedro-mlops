import numpy as np


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
