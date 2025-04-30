import numpy as np
import pytest

from kedro_mlops.library.evaluation import compute_lift


def test_compute_lift_perfect_prediction():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    lift_at = 0.6
    lift = compute_lift(y_true, y_pred, lift_at)
    assert lift > 1.0, "Lift should be greater than 1 for perfect predictions."


def test_compute_lift_no_negative_cases():
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    lift_at = 0.6
    lift = compute_lift(y_true, y_pred, lift_at)
    assert lift == 1.0, "Lift should be 1 when all cases are positive."


def test_compute_lift_invalid_lift_at():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    lift_at = 1.5  # Invalid lift_at value
    with pytest.raises(ValueError):
        compute_lift(y_true, y_pred, lift_at)


def test_compute_lift_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])
    lift_at = 0.5
    with pytest.raises(ValueError):
        compute_lift(y_true, y_pred, lift_at)
