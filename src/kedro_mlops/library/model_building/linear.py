import polars as pl
from kedro.utils import load_obj
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SequentialFeatureSelector

from kedro_mlops.library.utils import materialize_data


def sequential_feature_selection(
    train_data: pl.DataFrame,
    target: str,
    mod_params: dict,
) -> list:
    cnames = [cname for cname in train_data.columns if cname != target]

    model = load_obj(mod_params["model"]["class"])(**mod_params["model"]["kwargs"])

    sfs = SequentialFeatureSelector(
        model, **mod_params["sequential_feature_selection_kwargs"]
    )

    # cast target to pd.Series to avoid a warning/error on expecting a 1d array
    sfs.fit(
        train_data.select(cnames), train_data.select(target).to_series().to_pandas()
    )

    return sfs.get_feature_names_out()


def train_model(
    train_data: pl.DataFrame,
    target: str,
    selected_features: list,
    mod_params: dict,
) -> BaseEstimator:
    model = load_obj(mod_params["model"]["class"])(**mod_params["model"]["kwargs"])

    model.fit(train_data[selected_features], train_data[target])

    return model


def get_predictions(
    data: pl.DataFrame | pl.LazyFrame,
    selected_features: list,
    model: BaseEstimator,
    use_predict_proba: bool = True,
) -> pl.DataFrame:
    df = materialize_data(data)
    if use_predict_proba:
        y_pred = model.predict_proba(df[selected_features])[:, 1]
    else:
        y_pred = model.predict(df[selected_features])

    return df.with_columns(predictions=pl.Series("y_pred", y_pred))
