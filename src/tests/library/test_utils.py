import polars as pl
import pytest

from kedro_mlops.library.utils import materialize_data


@pytest.mark.parametrize("use_lazy_api", [False, True], ids=["eager", "lazy"])
def test_materialize_data(use_lazy_api: bool):
    data = pl.DataFrame({"col": [0, 1, 2]})
    if use_lazy_api:
        data = data.lazy()

    actual = materialize_data(data)
    assert isinstance(actual, pl.DataFrame)
