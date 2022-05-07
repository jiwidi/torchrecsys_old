from typing import List

##TODO
class feature:
    def __init__(self, name: str, dtype: str, unique_value_count: int) -> None:
        self.name = name
        self.dtype = dtype
        self.unique_value_count = unique_value_count
        self.layer_name = None

    def __str__(self):
        return f"Featue [Name: {self.name}, dtype: {self.dtype}, unique_value_count: {self.unique_value_count}]"


def dataframe_schema(df) -> List[feature]:
    r = []
    for col in df.columns.values:
        col_feature = feature(
            name=col, dtype=df[col].dtype.name, unique_value_count=len(df[col].unique())
        )
        r.append(col_feature)

    return r
