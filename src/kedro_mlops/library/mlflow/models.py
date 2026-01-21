from copy import copy
from pathlib import Path
from sys import version_info

from cloudpickle import __version__ as cloudpickle_version
from kedro import __version__ as kedro_version
from kedro.io import DataCatalog, MemoryDataset
from kedro.io.core import get_filepath_str
from kedro.utils import load_obj
from mlflow import __version__ as mlflow_version
from mlflow.pyfunc import PythonModel
from sklearn import __version__ as sklearn_version
from xgboost import __version__ as xgboost_version


class KedroModel(PythonModel):
    """A wrapper for a model trained with a kedro pipeline to be used with MLflow.

    This class is a subclass of `mlflow.pyfunc.PythonModel` because it also loads
    preprocessing artifacts and applies them to the input data before making the final
    predictions.
    """

    def __init__(self, pipeline, catalog):
        self.runner = None
        self.input_dataset_name = None
        self.pipeline = pipeline
        self.catalog = DataCatalog(
            datasets={name: MemoryDataset() for name in self.pipeline.datasets()}
        )
        self.catalog_artifacts = self._get_artifact_catalog(catalog)

        nb_outputs = len(self.pipeline.outputs())
        if nb_outputs != 1:
            outputs_list_str = "\n - ".join(self.pipeline.outputs())
            raise ValueError(
                f"Pipeline must have one and only one output, got '{nb_outputs}' outputs: \n - {outputs_list_str}"
            )
        self.output_name = next(iter(self.pipeline.outputs()))

    def get_artifacts(self):
        return {
            name: get_filepath_str(dataset._get_load_path(), dataset._protocol)  # noqa: SLF001
            for name, dataset in self.catalog_artifacts._datasets.items()  # noqa: SLF001
        }

    def predict(self, context, model_input, params=None):  # noqa: ARG002
        if params is None:
            params = {}

        runner_class = params.pop("runner", "SequentialRunner")

        # we don't want to recreate the runner object on each predict
        # because reimporting comes with a performance penalty in a serving setup
        # so if it is the default we just use the existing runner
        if self.runner is None or runner_class != type(self.runner).__name__:
            runner = load_obj(
                runner_class, "kedro.runner"
            )()  # do not forget to instantiate the class with ending ()

        # Register the runner if it is not already registered
        if self.runner is None:
            self.runner = runner

        self.catalog.save(
            ds_name=self.input_dataset_name,
            data=model_input,
        )

        run_output = runner.run(
            pipeline=self.pipeline,
            catalog=self.catalog,
            hook_manager=None,
        )

        # unpack the result to avoid messing the json
        # file with the name of the Kedro dataset
        unpacked_output = run_output[self.output_name]

        return unpacked_output

    def load_context(self, context):
        # a consistency check is made when loading the model
        # it would be better to check when saving the model
        # but we rely on a mlflow function for saving, and it is unaware of kedro
        # pipeline structure
        mlflow_artifacts_keys = set(context.artifacts.keys())
        kedro_artifacts_keys = set(self.catalog_artifacts._datasets.keys())  # noqa: SLF001
        if mlflow_artifacts_keys != kedro_artifacts_keys:
            in_artifacts_but_not_inference = (
                mlflow_artifacts_keys - kedro_artifacts_keys
            )
            in_inference_but_not_artifacts = (
                kedro_artifacts_keys - mlflow_artifacts_keys
            )
            raise ValueError(
                "Provided artifacts do not match catalog entries:"
                f"\n    - 'artifacts - inference.inputs()' = : {in_artifacts_but_not_inference}"
                f"\n    - 'inference.inputs() - artifacts' = : {in_inference_but_not_artifacts}"
            )

        updated_catalog = copy(self.catalog_artifacts)
        for name, uri in context.artifacts.items():
            updated_catalog._datasets[name]._filepath = Path(uri)  # noqa: SLF001
            self.catalog.save(ds_name=name, data=updated_catalog.load(name))

    def _get_artifact_catalog(self, catalog: DataCatalog) -> DataCatalog:
        """Get a catalog of artifacts to be logged to MLflow."""
        artifact_catalog = self._filter_catalog(catalog)
        self._verify_artifact_catalog(artifact_catalog)
        return artifact_catalog

    def _filter_catalog(self, catalog: DataCatalog) -> DataCatalog:
        """Filter the catalog to only include datasets that are used in the pipeline."""
        artifact_catalog = DataCatalog()
        for dataset_name, dataset in catalog._datasets.items():  # noqa: SLF001
            if dataset_name not in self.pipeline.datasets():
                continue

            metadata = dataset.metadata or {}
            if metadata.get("mlflow", {}).get("is_model_artifact", False):
                artifact_catalog[dataset_name] = dataset

        return artifact_catalog

    def _verify_artifact_catalog(self, catalog: DataCatalog) -> None:
        """Verify that the catalog contains all required datasets.

        The catalog should contain all datasets that are used in the inference pipeline.
        except for the input dataset, which is passed as an argument to the predict
        method.
        """
        missing_datasets = []
        for dataset_name in self.pipeline.inputs():
            if dataset_name.startswith("params:"):
                continue
            if dataset_name not in catalog._datasets:  # noqa: SLF001
                missing_datasets.append(dataset_name)

        if len(missing_datasets) == 1:
            self.input_dataset_name = missing_datasets[0]
        else:
            raise KedroModelError(
                f"There must be one and only one input dataset in the pipeline, "
                f"but got {len(missing_datasets)}: {missing_datasets}. "
            )

    @staticmethod
    def get_conda_env():
        """Get the dependencies for the model."""
        return {
            "channels": ["defaults"],
            "dependencies": [
                f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
                "pip",
                {
                    "pip": [
                        f"cloudpickle=={cloudpickle_version}",
                        f"kedro=={kedro_version}",
                        # only the skinny version of mlflow is needed
                        # because we do not need the tracking server or the REST API.
                        f"mlflow-skinny=={mlflow_version}",
                        f"scikit-learn=={sklearn_version}",
                        f"xgboost-cpu=={xgboost_version}",
                    ],
                },
            ],
            "name": "model_env",
        }


class KedroModelError(Exception):  # pragma: no cover
    """Error raised when the KedroModel construction fails"""
