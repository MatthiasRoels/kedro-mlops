from unittest.mock import MagicMock

import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro_datasets.pickle import PickleDataset

from src.kedro_mlops.library.mlflow.models import KedroModel


@pytest.fixture
def pipeline():
    pipeline = MagicMock(spec=Pipeline)
    pipeline.datasets.return_value = {"input_dataset", "model", "output_dataset"}
    pipeline.inputs.return_value = {"input_dataset", "model"}
    pipeline.outputs.return_value = {"output_dataset"}
    return pipeline


@pytest.fixture
def catalog(tmp_path):
    catalog = MagicMock(spec=DataCatalog)
    catalog._datasets = {
        "model": PickleDataset(
            filepath=(tmp_path / "encoder.pkl").resolve().as_posix(),
            metadata={"mlflow": {"is_model_artifact": True}},
        ),
        "params:example_param": MemoryDataset(),
    }
    return catalog


def test_kedro_model_initialization(pipeline, catalog):
    model = KedroModel(pipeline, catalog)

    assert model.pipeline == pipeline
    assert isinstance(model.catalog, DataCatalog)
    assert model.output_name == "output_dataset"


def test_kedro_model_initialization_invalid_outputs(pipeline, catalog):
    pipeline.outputs.return_value = {"output1", "output2"}

    with pytest.raises(ValueError, match="Pipeline must have one and only one output"):
        KedroModel(pipeline, catalog)


def test_kedro_model_get_artifacts(pipeline, catalog):
    model = KedroModel(pipeline, catalog)
    artifacts = model.get_artifacts()

    assert isinstance(artifacts, dict)
    assert "model" in artifacts


def test_kedro_model_load_context(mocker, tmp_path, pipeline, catalog):
    model = KedroModel(pipeline, catalog)
    mock_context = MagicMock()
    mock_context.artifacts = {"model": tmp_path / "encoder.pkl"}

    def mock_load(self):
        return "my-artifact"

    mocker.patch(
        "kedro_datasets.pickle.PickleDataset.load",
        mock_load,
    )
    model.load_context(mock_context)

    assert model.catalog.load("model") is not None


def test_kedro_model_predict():
    pass


def test_kedro_model_filter_catalog(pipeline, catalog):
    model = KedroModel(pipeline, catalog)
    filtered_catalog = model._filter_catalog(catalog)

    assert isinstance(filtered_catalog, DataCatalog)
    assert "model" in filtered_catalog._datasets


def test_kedro_model_get_conda_env():
    conda_env = KedroModel.get_conda_env()

    assert isinstance(conda_env, dict)
    assert "dependencies" in conda_env
    assert "pip" in conda_env["dependencies"]
    assert any("cloudpickle" in dep for dep in conda_env["dependencies"][2]["pip"])
