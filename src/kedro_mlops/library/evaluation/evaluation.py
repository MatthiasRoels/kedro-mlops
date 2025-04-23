import mlflow
import polars as pl
from mlflow.data.evaluation_dataset import EvaluationDataset

from kedro_mlops.library.evaluation.mlflow_evaluator import CustomModelEvaluator


def evaluate(
    model_type: str,
    model_outputs: pl.DataFrame,
    selected_features: list[str],
    target: str,
    predictions: str = "predictions",
):
    if mlflow.active_run() is not None:
        run_id = mlflow.active_run().info.run_id
        return _evaluate(
            data=model_outputs.to_pandas(),
            feature_names=selected_features,
            targets=target,
            predictions=predictions,
            model_type=model_type,
            run_id=run_id,
        )


# We cannot register custom ModelEvaluator (the only way is to create an MLflow plugin)
# which would be a bit overkill in this simple case. Hence we create a `_evaluate` method
# mimicking `mlflow.evaluate`, simplifying its logic drastically.
def _evaluate(  # pragma: no cover
    run_id,
    model=None,
    data=None,
    feature_names=None,
    targets=None,
    predictions=None,
    model_type=None,
):
    if targets is None:
        targets = "target"
    if predictions is None:
        predictions = "predictions"

    dataset = EvaluationDataset(
        data,
        targets=targets,
        feature_names=feature_names,
        predictions=predictions,
    )

    if CustomModelEvaluator.can_evaluate(model_type=model_type):
        evaluator = CustomModelEvaluator()

        evaluator.evaluate(
            model_type=model_type,
            dataset=dataset,
            run_id=run_id,
            model=model,
            predictions=predictions,
        )
