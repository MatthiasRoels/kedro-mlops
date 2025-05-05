import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from kedro_mlops.library.evaluation import (
    compute_classfier_metrics,
    compute_regressor_metrics,
)
from kedro_mlops.library.evaluation.plotting import (
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pr_curve,
    plot_predictions_vs_actuals,
    plot_qq,
    plot_roc_curve,
)


def evaluate(
    model_type: str,
    model_outputs: pl.DataFrame,
    selected_features: list[str],
    target: str,
    predictions: str = "predictions",
):
    """Evaluate the model using MLflow."""
    if model_type == "classifier":
        return _evaluate_classifier(
            model_outputs=model_outputs,
            selected_features=selected_features,
            target=target,
            predictions=predictions,
        )
    elif model_type == "regressor":
        return _evaluate_regressor(
            model_outputs=model_outputs,
            selected_features=selected_features,
            target=target,
            predictions=predictions,
        )
    else:
        raise ValueError(f"Unsupported model type {model_type}")


def _evaluate_classifier(
    model_outputs: pl.DataFrame,
    selected_features: list[str],
    target: str,
    predictions: str,
    threshold: float = 0.5,
):
    y_true = model_outputs.filter(pl.col("split") == "test").select(target).to_numpy()
    y_pred = (
        model_outputs.filter(pl.col("split") == "test").select(predictions).to_numpy()
    )
    y_pred_b = np.array([0 if pred <= threshold else 1 for pred in y_pred])

    metrics = compute_classfier_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_b=y_pred_b,
    )

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)

    # return artifacts as matplotlib figures
    artifacts = {
        "roc-curve": plot_roc_curve(fpr=fpr, tpr=tpr, auc=metrics["AUC"]),
        "precision-recall-curve": plot_pr_curve(
            precision=precision,
            recall=recall,
            average=metrics["Average Precision score"],
        ),
        "confusion-matrix": plot_confusion_matrix(
            confusion_matrix=confusion_matrix(y_true, y_pred_b)
        ),
        "correlation-matrix": plot_correlation_matrix(
            df_corr=model_outputs.select(selected_features).to_pandas().corr()
        ),
    }

    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)

        for plot_name, plot in artifacts.items():
            mlflow.log_figure(figure=plot, artifact_file=f"{plot_name}.png")

    return artifacts


def _evaluate_regressor(
    model_outputs: pl.DataFrame,
    selected_features: list[str],
    target: str,
    predictions: str = "predictions",
):
    y_true = model_outputs.filter(pl.col("split") == "test").select(target).to_pandas()
    y_pred = (
        model_outputs.filter(pl.col("split") == "test").select(predictions).to_pandas()
    )

    metrics = compute_regressor_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    artifacts = {
        "predictions-vs-actuals": plot_predictions_vs_actuals(
            y_true=y_true, y_pred=y_pred
        ),
        "qq-plot": plot_qq(y_true=y_true, y_pred=y_pred),
        "correlation-matrix": plot_correlation_matrix(
            df_corr=model_outputs.select(selected_features).to_pandas().corr()
        ),
    }

    # log metrics
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)

        for plot_name, plot in artifacts.items():
            mlflow.log_figure(figure=plot, artifact_file=f"{plot_name}.png")

    # return artifacts as matplotlib figures
    return artifacts
