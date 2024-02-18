import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError

from src.kedro_mlops.library.preprocessing.target_encoder import TargetEncoder


class TestTargetEncoder:  # noqa: D101
    def test_target_encoder_constructor_weight_value_error(self):
        with pytest.raises(ValueError):
            TargetEncoder(weight=-1)

    def test_target_encoder_constructor_imputation_value_error(self):
        with pytest.raises(ValueError):
            TargetEncoder(imputation_strategy="median")

    # Tests for fit method
    def test_target_encoder_fit_binary_classification(self):
        # test_target_encoder_fit_column_linear_regression() tested on one
        # column input as a numpy series; this test runs on a dataframe input.
        df = pl.DataFrame(
            {
                "variable": [
                    "positive",
                    "positive",
                    "negative",
                    "neutral",
                    "negative",
                    "positive",
                    "negative",
                    "neutral",
                    "neutral",
                    "neutral",
                ],
                "target": [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            }
        )

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")

        expected = {
            "negative": 0.333333,
            "neutral": 0.50000,
            "positive": 0.666667,
        }
        actual = encoder.mapping_["variable"]

        assert actual == pytest.approx(expected, rel=1e-3, abs=1e-3)

    def test_target_encoder_fit_linear_regression(self):
        # test_target_encoder_fit_column_linear_regression() tested on one
        # column input as a numpy series; this test runs on a dataframe input.
        df = pl.DataFrame(
            {
                "variable": [
                    "positive",
                    "positive",
                    "negative",
                    "neutral",
                    "negative",
                    "positive",
                    "negative",
                    "neutral",
                    "neutral",
                    "neutral",
                    "positive",
                ],
                "target": [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4],
            }
        )

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")

        expected = {
            "negative": -4.666667,
            "neutral": 0.250000,
            "positive": 4.500000,
        }

        actual = encoder.mapping_["variable"]

        assert actual == pytest.approx(expected, rel=1e-3, abs=1e-3)

    # Tests for transform method
    def test_target_encoder_transform_when_not_fitted(self):
        df = pl.DataFrame(
            {
                "variable": [
                    "positive",
                    "positive",
                    "negative",
                    "neutral",
                    "negative",
                    "positive",
                    "negative",
                    "neutral",
                    "neutral",
                    "neutral",
                ],
                "target": [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            }
        )

        encoder = TargetEncoder()
        with pytest.raises(NotFittedError):
            encoder.transform(data=df)

    def test_target_encoder_transform_binary_classification(self):
        df = pl.DataFrame(
            {
                "variable": [
                    "positive",
                    "positive",
                    "negative",
                    "neutral",
                    "negative",
                    "positive",
                    "negative",
                    "neutral",
                    "neutral",
                    "neutral",
                ],
                "target": [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            }
        )

        values = [
            0.666667,
            0.666667,
            0.333333,
            0.50000,
            0.333333,
            0.666667,
            0.333333,
            0.50000,
            0.50000,
            0.50000,
        ]
        expected = df.clone().with_columns(
            pl.Series(name="variable_enc", values=values)
        )

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")
        actual = encoder.transform(data=df)

        assert_frame_equal(actual, expected)

    def test_target_encoder_transform_linear_regression(self):
        df = pl.DataFrame(
            {
                "variable": [
                    "positive",
                    "positive",
                    "negative",
                    "neutral",
                    "negative",
                    "positive",
                    "negative",
                    "neutral",
                    "neutral",
                    "neutral",
                    "positive",
                ],
                "target": [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4],
            }
        )

        values = [
            4.500000,
            4.500000,
            -4.666667,
            0.250000,
            -4.666667,
            4.500000,
            -4.666667,
            0.250000,
            0.250000,
            0.250000,
            4.500000,
        ]

        expected = df.clone().with_columns(
            pl.Series(name="variable_enc", values=values)
        )

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")
        actual = encoder.transform(data=df)

        assert_frame_equal(actual, expected)

    # def test_target_encoder_transform_new_category_binary_classification(self):
    #     df = pl.DataFrame(
    #         {
    #             "variable": [
    #                 "positive",
    #                 "positive",
    #                 "negative",
    #                 "neutral",
    #                 "negative",
    #                 "positive",
    #                 "negative",
    #                 "neutral",
    #                 "neutral",
    #                 "neutral",
    #             ],
    #             "target": [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    #         }
    #     )

    #     df_appended = df.append({"variable": "new", "target": 1}, ignore_index=True)

    #     # inputs of TargetEncoder will be of dtype category
    #     df["variable"] = df["variable"].astype("category")
    #     df_appended["variable"] = df_appended["variable"].astype("category")

    #     expected = df_appended.copy()
    #     expected["variable_enc"] = [
    #         0.666667,
    #         0.666667,
    #         0.333333,
    #         0.50000,
    #         0.333333,
    #         0.666667,
    #         0.333333,
    #         0.50000,
    #         0.50000,
    #         0.50000,
    #         0.333333,
    #     ]

    #     encoder = TargetEncoder(imputation_strategy="min")
    #     encoder.fit(data=df, column_names=["variable"], target_column="target")
    #     actual = encoder.transform(data=df_appended, column_names=["variable"])

    #     pd.testing.assert_frame_equal(actual, expected)

    # def test_target_encoder_transform_new_category_linear_regression(self):
    #     df = pl.DataFrame({'variable': ['positive', 'positive', 'negative',
    #                                     'neutral', 'negative', 'positive',
    #                                     'negative', 'neutral', 'neutral',
    #                                     'neutral', 'positive'],
    #                        'target': [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]})

    #     df_appended = df.append({"variable": "new", "target": 10},
    #                             ignore_index=True)

    #     # inputs of TargetEncoder will be of dtype category
    #     df["variable"] = df["variable"].astype("category")
    #     df_appended["variable"] = df_appended["variable"].astype("category")

    #     expected = df_appended.copy()
    #     expected["variable_enc"] = [
    #         4.500000,
    #         4.500000,
    #         -4.666667,
    #         0.250000,
    #         -4.666667,
    #         4.500000,
    #         -4.666667,
    #         0.250000,
    #         0.250000,
    #         0.250000,
    #         4.500000,
    #         -4.666667,
    #     ]  # min imputation for new value

    #     encoder = TargetEncoder(imputation_strategy="min")
    #     encoder.fit(data=df, column_names=["variable"], target_column="target")
    #     actual = encoder.transform(data=df_appended, column_names=["variable"])

    #     pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "cname, expected",
        [
            ("test_column", "test_column_enc"),
            ("test_column_bin", "test_column_enc"),
            ("test_column_processed", "test_column_enc"),
            ("test_column_cleaned", "test_column_enc"),
        ],
    )
    def test_target_encoder_clean_column_name_binned_column(self, cname, expected):
        encoder = TargetEncoder()
        actual = encoder._clean_column_name(cname)

        assert actual == expected
