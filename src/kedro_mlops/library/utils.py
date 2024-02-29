import polars as pl


def materialize_data(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data
