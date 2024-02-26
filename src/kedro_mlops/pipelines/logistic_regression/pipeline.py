"""This is a boilerplate pipeline 'logistic_regression'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from kedro_mlops.library.preprocessing.nodes import (
    apply_variance_threshold,
    fit_discretizer,
    fit_encoder,
    stratified_train_test_split_binary_target,
    transform_data,
)


def create_pipeline(**kwargs) -> Pipeline:  # pragma: no cover
    return pipeline(
        [
            node(
                func=stratified_train_test_split_binary_target,
                inputs=[
                    "input_features",
                    "params:input_data_schema.target",
                    "params:preprocessing.train_test_split.test_size",
                ],
                outputs="train_test_set",
                name="stratified_train_test_split_node",
            ),
            node(
                func=apply_variance_threshold,
                inputs=[
                    "train_test_set",
                    "params:preprocessing.variance_threshold.threshold",
                ],
                outputs="filtered_train_test_set",
                name="variance_threshold_node",
            ),
            node(
                func=fit_discretizer,
                inputs=[
                    "filtered_train_test_set",
                    "params:input_data_schema.numeric_columns",
                    "params:preprocessing.kbins_discretizer",
                ],
                outputs="fitted_discretizer",
                name="fit_discretizer_node",
            ),
            node(
                func=transform_data,
                inputs=["filtered_train_test_set", "fitted_discretizer"],
                outputs="discretized_data",
                name="discretize_data_node",
            ),
            node(
                func=fit_encoder,
                inputs=[
                    "discretized_data",
                    "params:input_data_schema.numeric_columns",
                    "params:input_data_schema.target",
                    "params:preprocessing.target_encoder",
                ],
                outputs="fitted_encoder",
                name="fit_encoder_node",
            ),
            node(
                func=transform_data,
                inputs=[
                    "discretized_data",
                    "fitted_encoder",
                ],
                outputs="preprocessed_data",
                name="encode_data_node",
            ),
        ]
    )
