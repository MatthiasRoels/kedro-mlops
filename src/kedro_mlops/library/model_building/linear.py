import polars as pl
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from kedro_mlops.library.utils import materialize_data


def feature_selection(
    train_data: pl.DataFrame,
    target: str,
) -> list:
    cnames = [cname for cname in train_data.columns if cname != target]

    logit = LogisticRegression(
        fit_intercept=True,
        C=1e9,
        solver='liblinear',
        random_state=42
    )

    sfs = SequentialFeatureSelector(
        logit,
        tol=10e-3,
        direction="forward",
        scoring="roc_auc",
        cv=5,
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
) -> LogisticRegression:

    logit = LogisticRegression(
        fit_intercept=True,
        C=1e9,
        solver='liblinear',
        random_state=42
    )

    logit.fit(train_data[selected_features], train_data[target])

    return logit


def get_predictions(
    data: pl.DataFrame | pl.LazyFrame,
    selected_features: list,
    logit: LogisticRegression,
) -> pl.DataFrame:
    df = materialize_data(data)
    y_pred = logit.predict_proba(df[selected_features])[:, 1]
    return df.with_columns(predictions=pl.Series("y_pred", y_pred))
