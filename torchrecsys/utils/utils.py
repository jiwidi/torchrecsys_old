import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


class feature_catalog:
    def __init__(self, data: pd.DataFrame, id_column_name: str):
        self.data = data
        self.id_column_name = id_column_name
        # dataset has one row per id
        assert len(data) == len(data.drop_duplicates(subset=[id_column_name]))

        columns = self.data.columns.values.tolist()
        self.categorical_columns = self.data.select_dtypes(
            include="category"
        ).columns.values.tolist()  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
        self.categorical_columns_idx = [
            columns.index(x) for x in self.categorical_columns
        ]

        self.continous_columns = self.data.select_dtypes(
            include="number"
        ).columns.values.tolist()
        self.continous_columns_idx = [columns.index(x) for x in self.continous_columns]

        # ZIPEAR columns con su idx tmb para las transforms despues
        # Encode all categorical columns
        self.label_encoders = {}
        for categorical_column in self.categorical_columns:
            aux = LabelEncoder()

            self.label_encoders[categorical_column] = aux
        # Build the catalog
        self.categorical_catalog = {}
        self.continous_catalog = {}

        # TODO Optim this
        for idx, row in tqdm(
            self.data.iterrows(), total=len(self.data), desc="Building catalog"
        ):
            self.categorical_catalog[row[id_column_name]] = row[
                self.categorical_columns
            ].values
            self.continous_catalog[row[id_column_name]] = row[
                self.continous_columns
            ].values

    def __getitem__(self, key):
        categorical = self.categorical_catalog[key]
        continous = self.continous_catalog[key]

        return {
            "categorical": categorical,
            "continous": continous,
        }
