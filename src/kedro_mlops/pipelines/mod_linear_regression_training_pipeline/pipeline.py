"""This is a boilerplate pipeline 'logistic_regression'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from kedro_mlops.library.evaluation.nodes import evaluate
from kedro_mlops.library.model_building.linear import (
    get_predictions,
    sequential_feature_selection,
    train_model,
)
from kedro_mlops.library.preprocessing.nodes import (
    apply_variance_threshold,
    fit_discretizer,
    fit_encoder,
    prepare_train_data,
    transform_data,
)
from kedro_mlops.library.preprocessing.utils import (
    train_test_split,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return pipeline(
        [
            node(
                func=train_test_split,
                inputs=[
                    "params:mod_params.model.model_type",
                    "input_features",
                    "params:mod_params.preprocessing.train_test_split.test_size",
                    "params:input_data_schema.target",
                ],
                outputs="train_test_set",
                name="train_test_split_node",
            ),
            node(
                func=apply_variance_threshold,
                inputs=[
                    "train_test_set",
                    "params:input_data_schema.target",
                    "params:mod_params.preprocessing.variance_threshold.threshold",
                ],
                outputs="filtered_input_features",
                name="apply_variance_threshold_node",
            ),
            node(
                func=fit_discretizer,
                inputs=[
                    "filtered_input_features",
                    "params:input_data_schema.numeric_columns",
                    "params:mod_params.preprocessing.kbins_discretizer",
                ],
                outputs="fitted_discretizer",
                name="fit_kbins_discretizer_node",
            ),
            node(
                func=transform_data,
                inputs=["filtered_input_features", "fitted_discretizer"],
                outputs="discretized_data",
                name="discretize_data_node",
                tags=["inference"],
            ),
            node(
                func=fit_encoder,
                inputs=[
                    "discretized_data",
                    "params:input_data_schema.target",
                    "params:mod_params.preprocessing.target_encoder",
                    "params:input_data_schema.pk_col",
                ],
                outputs="fitted_encoder",
                name="fit_target_encoder_node",
            ),
            node(
                func=transform_data,
                inputs=[
                    "discretized_data",
                    "fitted_encoder",
                ],
                outputs="preprocessed_data",
                name="target_encode_data_node",
                tags=["inference"],
            ),
            node(
                func=prepare_train_data,
                inputs=[
                    "preprocessed_data",
                    "params:input_data_schema.target",
                    "params:mod_params.preprocessing.univariate_feature_selection.threshold",
                    "params:mod_params.model.model_type",
                ],
                outputs="training_data",
                name="univariate_feature_selection_node",
            ),
            node(
                func=sequential_feature_selection,
                inputs=[
                    "training_data",
                    "params:input_data_schema.target",
                    "params:mod_params",
                ],
                outputs="selected_features",
                name="feature_selection_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "training_data",
                    "params:input_data_schema.target",
                    "selected_features",
                    "params:mod_params",
                ],
                outputs="fitted_regression_model",
                name="train_model_node",
            ),
            node(
                func=get_predictions,
                inputs=[
                    "preprocessed_data",
                    "fitted_regression_model",
                ],
                outputs="model_outputs",
                name="get_predictions_node",
                tags=["inference"],
            ),
            node(
                func=evaluate,
                inputs=[
                    "params:mod_params.model.model_type",
                    "model_outputs",
                    "selected_features",
                    "params:input_data_schema.target",
                ],
                outputs="mlflow_plots",
                name="evaluate_model",
            ),
        ],
        tags=["training", "linear_regression"],
    )
