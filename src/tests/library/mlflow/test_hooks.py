from unittest.mock import MagicMock, patch

import pytest
from kedro.framework.context import KedroContext

from kedro_mlops.library.mlflow.hooks import MlflowHook


@pytest.fixture
def mock_context(tmp_path):
    context = MagicMock(spec=KedroContext)
    context.config_loader = {
        "mlflow": {
            "server": {"tracking_uri": None},
            "tracking": {
                "enable_tracking": True,
                "exclude_pipelines": ["excluded_pipeline"],
                "experiment": {"name": "test_experiment", "tags": []},
                "run": {"id": None, "name": "test_run"},
            },
        }
    }
    context.project_path = tmp_path
    return context


@pytest.fixture
def mlflow_hook():
    return MlflowHook()


@patch("mlflow.MlflowClient")
@patch("mlflow.set_tracking_uri")
@patch("mlflow.autolog")
def test_after_context_created(
    mock_autolog, mock_set_tracking_uri, mock_mlflow_client, mlflow_hook, mock_context
):
    mlflow_hook.after_context_created(mock_context)

    mock_set_tracking_uri.assert_called_once()
    mock_autolog.assert_called_once_with(disable=True)
    assert mlflow_hook._mlflow_client is not None
    assert not mlflow_hook._mlflow_disabled


def test_after_context_created_no_config(mlflow_hook):
    mock_context = MagicMock(spec=KedroContext)
    mock_context.config_loader = {"mlflow": {}}

    mlflow_hook.after_context_created(mock_context)

    assert mlflow_hook._mlflow_disabled


@patch("mlflow.start_run")
@patch("mlflow.active_run")
def test_before_pipeline_run(
    mock_active_run, mock_start_run, mlflow_hook, mock_context
):
    mlflow_hook.after_context_created(mock_context)
    run_params = {"pipeline_name": "test_pipeline"}

    mlflow_hook.before_pipeline_run(run_params)

    mock_start_run.assert_called_once()
    assert mlflow_hook.run_id is not None


@patch("mlflow.start_run")
@patch("mlflow.active_run")
def test_before_node_run(mock_active_run, mock_start_run, mlflow_hook, mock_context):
    mlflow_hook.after_context_created(mock_context)
    mlflow_hook.run_id = "test_run_id"

    mlflow_hook.before_node_run()

    mock_start_run.assert_called_once_with(run_id="test_run_id", nested=True)


@patch("mlflow.end_run")
def test_after_pipeline_run(mock_end_run, mlflow_hook, mock_context):
    mlflow_hook.after_context_created(mock_context)
    mlflow_hook.run_id = "test_run_id"

    mlflow_hook.after_pipeline_run(None, None)

    mock_end_run.assert_called_once()


@patch("mlflow.MlflowClient.get_experiment_by_name")
@patch("mlflow.MlflowClient.create_experiment")
def test_set_mlflow_experiment(
    mock_create_experiment, mock_get_experiment_by_name, mlflow_hook, mock_context
):
    mock_get_experiment_by_name.return_value = None
    mlflow_hook.after_context_created(mock_context)

    experiment_id = mlflow_hook._set_mlflow_experiment()

    mock_create_experiment.assert_called_once()
    assert experiment_id is not None


def test_assert_mlflow_enabled(mlflow_hook, mock_context):
    mlflow_hook.after_context_created(mock_context)

    assert mlflow_hook._assert_mlflow_enabled(
        "test_pipeline", mlflow_hook.mlflow_config
    )
    assert not mlflow_hook._assert_mlflow_enabled(
        "excluded_pipeline", mlflow_hook.mlflow_config
    )
