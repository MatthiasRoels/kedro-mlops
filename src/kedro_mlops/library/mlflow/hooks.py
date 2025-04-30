import logging
from pathlib import Path

import mlflow
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from mlflow.entities import RunStatus

from kedro_mlops.library.mlflow.schemas import MLflowConfig

logger = logging.getLogger(__name__)


class MlflowHook:
    """Hook class for MLflow.
    Configures mlflow server, setup experiment and runs etc.
    """

    def __init__(self):
        self.mlflow_config: MLflowConfig | None = None
        self.run_id: str | None = None
        self._mlflow_client: mlflow.MlflowClient | None = None
        self._mlflow_disabled: bool = False
        self._params = None

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        """Hooks to be invoked after a `KedroContext` is created. This is the earliest
        hook triggered within a Kedro run. The `KedroContext` stores useful information
        such as `credentials`, `config_loader` and `env`.
        Args:
            context: The context that was created.
        """
        mlflow_config = context.config_loader["mlflow"]

        if mlflow_config:
            self.mlflow_config = MLflowConfig.model_validate(mlflow_config)

            tracking_uri = self.mlflow_config.server.tracking_uri

            if tracking_uri is None:
                tracking_uri = (Path(context.project_path) / "data" / "mlruns").as_uri()

            self._mlflow_client = mlflow.MlflowClient(
                tracking_uri=tracking_uri,
                registry_uri=tracking_uri,
            )
            # IMPORTANT: Explicitely set tracking URI to be able to use e.g.
            # mlflow.start_run(...) correctly!!!
            mlflow.set_tracking_uri(tracking_uri)
            # Disable autolog as we favour custom logging
            mlflow.autolog(disable=True)
        else:
            self._mlflow_disabled = True
            logger.warning("No MLflow config found, disabling tracking ...")

    @hook_impl
    def before_pipeline_run(self, run_params: dict) -> None:
        """Hook to be invoked before a pipeline runs"""
        if self._mlflow_disabled:
            return None

        pipeline_name = run_params.get("pipeline_name")

        if pipeline_name is None or not self._assert_mlflow_enabled(
            pipeline_name, self.mlflow_config
        ):
            # Disable MLflow for particular pipelines
            self._mlflow_disabled = True
            return None

        # Set experiment and run
        experiment_id = self._set_mlflow_experiment()

        run_id = None
        run_name = pipeline_name
        if self.mlflow_config.tracking.run is not None:
            run_id = self.mlflow_config.tracking.run.id
            run_name = self.mlflow_config.tracking.run.name

        mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=True,
        )
        self.run_id = mlflow.active_run().info.run_id
        logger.info(
            f"Mlflow run '{mlflow.active_run().info.run_name}' - '{self.run_id}' has started"
        )

    @hook_impl
    def before_node_run(self) -> None:
        """Hook to be invoked before a node runs."""
        if not self._mlflow_disabled and self.run_id is not None:
            # Reopening the run ensures the run_id started at the beginning of the pipeline
            # is used for all tracking. This is necessary because to bypass mlflow thread safety
            # each call to the "active run" now creates a new run when started in a new thread. See
            # https://github.com/Galileo-Galilei/kedro-mlflow/issues/613
            # https://github.com/Galileo-Galilei/kedro-mlflow/pull/615
            # https://github.com/Galileo-Galilei/kedro-mlflow/issues/623
            # https://github.com/Galileo-Galilei/kedro-mlflow/issues/624
            try:
                mlflow.start_run(
                    run_id=self.run_id,
                    nested=True,
                )
                logger.info(
                    f"Restarting mlflow run '{mlflow.active_run().info.run_name}' - '{self.run_id}' at node level for multi-threading"
                )
            except Exception as err:  # pragma: no cover
                if f"Run with UUID {self.run_id} is already active" in str(err):
                    # This means that the run was started before in the same thread, likely at the beginning of another node
                    pass
                else:
                    raise err
            logger.info(f"Active run: {mlflow.active_run().info}")

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook to be invoked after a pipeline runs."""
        if not self._mlflow_disabled and self.run_id is not None:
            mlflow.end_run()

    @hook_impl
    def on_pipeline_error(self) -> None:
        """Hook to be invoked after a pipeline runs."""
        if not self._mlflow_disabled and self.run_id is not None:
            # first, close all runs within the thread
            while mlflow.active_run():
                current_run_id = mlflow.active_run().info.run_id
                logger.info(
                    f"The run '{current_run_id}' was closed because of an error in the pipeline."
                )
                mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
                pipeline_run_id_is_closed = current_run_id == self.run_id

            # second, ensure that parent run in another thread is closed
            if not pipeline_run_id_is_closed:
                self._mlflow_client.set_terminated(
                    self.run_id, RunStatus.to_string(RunStatus.FAILED)
                )
                logger.info(
                    f"The parent run '{self.run_id}' was closed because of an error in the pipeline."
                )

    def _set_mlflow_experiment(self) -> str:
        mlflow_experiment = self._mlflow_client.get_experiment_by_name(
            name=self.mlflow_config.tracking.experiment.name
        )

        if mlflow_experiment is None:
            experiment_id = self._mlflow_client.create_experiment(
                name=self.mlflow_config.tracking.experiment.name,
                tags=self.mlflow_config.tracking.experiment.tags,
            )
        else:
            experiment_id = mlflow_experiment.experiment_id

        return experiment_id

    @staticmethod
    def _assert_mlflow_enabled(pipeline_name: str, mlflow_config: MLflowConfig) -> bool:
        enable = mlflow_config.tracking.enable_tracking

        if pipeline_name in mlflow_config.tracking.exclude_pipelines:
            enable = False

        return enable
