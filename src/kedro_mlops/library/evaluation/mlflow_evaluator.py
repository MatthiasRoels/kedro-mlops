import logging
from pathlib import Path

import mlflow
import numpy as np
from mlflow.models.evaluation import (
    EvaluationResult,
    ModelEvaluator,
)
from mlflow.models.evaluation.artifacts import ImageEvaluationArtifact
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from kedro_mlops.library.evaluation.compute import compute_lift
from kedro_mlops.library.evaluation.plotting import (
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


class CustomModelEvaluator(ModelEvaluator):
    """Custom ModelEvaluator class tailored to our needs

    Custom class is needed to be able to use predictions generated with `predict_proba`.
    This allows to compute metrics and artifacts from a static dataset and
    a custom pyfunc model.
    """

    @classmethod
    def can_evaluate(cls, *, model_type, **kwargs):  # noqa: ARG003
        """Check if the evaluator can evaluate the specified model type

        Args:
            model_type: A string describing the model type (e.g., "regressor", "classifier", …).
            kwargs: unused args for compatibility with the metaclass.

        Returns:
            True if the evaluator can evaluate the specified model on the
            specified dataset. False otherwise.
        """
        return model_type in ["classifier", "regressor"]

    def evaluate(
        self,
        *,
        model_type,
        dataset,
        run_id,
        model,
        extra_metrics=None,
        custom_artifacts=None,
        predictions=None,
        **kwargs,  # noqa: ARG002
    ) -> EvaluationResult:
        """Computes and logs metrics and artifacts, and return evaluation results.

        Parameters
        ----------
            model_type: A string describing the model type
                (e.g., ``"regressor"``, ``"classifier"``, …).
            dataset: An instance of `mlflow.models.evaluation.base._EvaluationDataset`
                containing features and labels (optional) for model evaluation.
            run_id: The ID of the MLflow Run to which to log results.
            evaluator_config: A dictionary of additional configurations for
                the evaluator.
            model: A pyfunc model instance. If None, the model output is supposed to be found in
                ``dataset.predictions_data``.
            extra_metrics: A list of :py:class:`EvaluationMetric` objects.
            custom_artifacts: A list of callable custom artifact functions.
            predictions: The column name of the model output column that is used for evaluation.
                This is only used when a model returns a pandas dataframe that contains
                multiple columns.
            kwargs: For forwards compatibility, a placeholder for additional arguments that
                may be added to the evaluation interface in the future.

        Returns
        -------
            A :py:class:`mlflow.models.EvaluationResult` instance containing
            evaluation metrics and artifacts for the model.
        """
        X = dataset.features_data
        y_true = dataset.labels_data
        if model is not None:
            y_pred = model.predict(X)
        elif predictions is not None:
            y_pred = dataset.predictions_data
        else:
            raise ValueError("dataset and model argument cannot be both None.")

        if model_type == "classifier":
            return self._evaluate_classifier(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                run_id=run_id,
                extra_metrics=extra_metrics,
                custom_artifacts=custom_artifacts,
            )
        elif model_type == "regressor":
            return self._evaluate_regressor(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                run_id=run_id,
                extra_metrics=extra_metrics,
                custom_artifacts=custom_artifacts,
            )
        else:
            raise ValueError(f"Unsupported model type {model_type}")

    def _evaluate_classifier(
        self,
        *,
        X,
        y_true,
        y_pred,
        run_id,
        **kwargs,  # noqa: ARG002
    ) -> EvaluationResult:
        """Compute metrics and artifacts for a classfier"""
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_pred)
        precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)

        y_pred_b = np.array([0 if pred <= 0.5 else 1 for pred in y_pred])

        metrics = self._compute_classfier_metrics(y_true, y_pred, y_pred_b)
        mlflow.log_metrics(metrics)
        # default artifacts:
        artifacts_dir = "data/artifacts"

        artifacts = {
            "roc-curve": self._log_image_artifact(
                plot_roc_curve,
                kwargs={"fpr": fpr, "tpr": tpr, "auc": metrics["AUC"]},
                artifact_name="roc-curve",
                artifacts_dir=artifacts_dir,
            ),
            "precision-recall-curve": self._log_image_artifact(
                plot_pr_curve,
                kwargs={
                    "precision": precision,
                    "recall": recall,
                    "average": metrics["Average Precision score"],
                },
                artifact_name="pr-curve",
                artifacts_dir=artifacts_dir,
            ),
            "confusion-matrix": self._log_image_artifact(
                plot_confusion_matrix,
                kwargs={"confusion_matrix": confusion_matrix(y_true, y_pred_b)},
                artifact_name="confusion-matrix",
                artifacts_dir=artifacts_dir,
            ),
            "correlation-matrix": self._log_image_artifact(
                plot_correlation_matrix,
                kwargs={"df_corr": X.corr()},
                artifact_name="correlation-matrix",
                artifacts_dir=artifacts_dir,
            ),
        }

        return EvaluationResult(metrics=metrics, artifacts=artifacts, run_id=run_id)

    def _evaluate_regressor(
        self,
        *,
        y_true,
        y_pred,
        run_id,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> EvaluationResult:
        """Compute metrics and artifacts for a regressor"""
        metrics = self._compute_regressor_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)

    @staticmethod
    def _compute_classfier_metrics(y_true, y_pred, y_pred_b) -> dict[str, float]:
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

    @staticmethod
    def _compute_regressor_metrics(y_true, y_pred) -> dict[str, float]:
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

    @staticmethod
    def _log_image_artifact(
        plot_fn,
        kwargs,
        artifact_name: str,
        artifacts_dir: str,
        metric_prefix: str | None = None,
    ):
        prefix = metric_prefix if metric_prefix else ""
        artifact_file_name = f"{prefix}{artifact_name}.png"
        artifact_file_local_path = Path(artifacts_dir) / artifact_file_name

        try:
            plot_fn(**kwargs, path=artifact_file_local_path)
        except Exception:
            logger.exception(f"Failed to log image artifact {artifact_name!r}")
        else:
            mlflow.log_artifact(artifact_file_local_path)
            artifact = ImageEvaluationArtifact(
                uri=mlflow.get_artifact_uri(artifact_file_name)
            )
            artifact._load(artifact_file_local_path)  # noqa: SLF001
            return artifact
