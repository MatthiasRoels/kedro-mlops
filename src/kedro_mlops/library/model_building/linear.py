import mlflow
import polars as pl
from kedro.utils import load_obj
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SequentialFeatureSelector


def sequential_feature_selection(
    train_data: pl.DataFrame,
    target: str,
    mod_params: dict,
) -> list:
    cnames = [cname for cname in train_data.columns if cname != target]

    model = load_obj(mod_params["model"]["class"])(**mod_params["model"]["kwargs"])

    sfs = SequentialFeatureSelector(model, **mod_params["feature_selection"]["kwargs"])

    # cast target to pd.Series to avoid a warning/error on expecting a 1d array
    sfs.fit(
        train_data.select(cnames), train_data.select(target).to_series().to_pandas()
    )

    feature_names = sfs.get_feature_names_out()

    if mlflow.active_run() is not None:  # pragma: no cover
        mlflow.log_params(
            {
                "Model": mod_params["model"]["class"],
                "selected_features": feature_names,
                "feature selection method": mod_params["feature_selection"]["method"],
            }
        )

    return feature_names


def train_model(
    train_data: pl.DataFrame,
    target: str,
    selected_features: list,
    mod_params: dict,
) -> BaseEstimator:
    model = load_obj(mod_params["model"]["class"])(**mod_params["model"]["kwargs"])

    model.fit(train_data.select(selected_features), train_data.select(target))

    # Remark: because we use a DataFrame as input for fitting the model, the
    # feature names are added to the `feature_names_in_` attribute of the model.
    # Hence, we can use this attribute to get the feature names when doing inference.
    return model


def get_predictions(
    data: pl.DataFrame | pl.LazyFrame,
    model: BaseEstimator,
) -> pl.DataFrame:
    selected_features = model.feature_names_in_
    df = data.lazy().collect()
    y_pred = model.predict_proba(df.select(selected_features))[:, 1]

    return df.with_columns(predictions=pl.Series("y_pred", y_pred))
