# Arrange
import pytest
import numpy as np
import pandas as pd
from torchrecsys.datasets import InteractionsDataset


@pytest.fixture
def dummy_interactions():
    interactions = np.array(
        [[1, 1, 5], [1, 2, 3], [1, 3, 3], [2, 2, 4], [2, 3, 5], [2, 4, 4], [2, 5, 5],],
        dtype=int,
    )
    interactions = pd.DataFrame(interactions, columns=["user_id", "item_id", "rating"],)
    return interactions


@pytest.fixture
def dummy_user_features():
    users = np.array(
        [
            [1, "F", 1, "10", "48067"],
            [2, "M", 56, "16", "70072"],
            [3, "M", 25, "15", "55117"],
            [4, "M", 45, "7", "02460"],
            [5, "M", 25, "20", "55455"],
        ]
    )
    users = pd.DataFrame(
        users, columns=["user_id", "gender", "age", "occupation", "zip"]
    )

    # Preprocess users
    users["user_id"] = pd.to_numeric(users["user_id"])
    users["age"] = pd.to_numeric(users["age"])

    users["gender"], uniques = pd.factorize(users["gender"])
    users["occupation"], uniques = pd.factorize(users["occupation"])
    users["zip"], uniques = pd.factorize(users["zip"])
    ##Set category dtype
    users["gender"] = users.gender.astype("category")
    users["occupation"] = users.occupation.astype("category")
    users["zip"] = users.zip.astype("category")

    return users


@pytest.fixture
def dummy_item_features():
    items = np.array(
        [
            [1, "Toy Story (1995)", "Animation|Children's|Comedy"],
            [2, "Jumanji (1995)", "Adventure|Children's|Fantasy"],
            [3, "Grumpier Old Men (1995)", "Comedy|Romance"],
            [4, "Waiting to Exhale (1995)", "Comedy|Drama"],
            [5, "Father of the Bride Part II (1995)", "Comedy"],
        ]
    )
    items = pd.DataFrame(items, columns=["item_id", "title", "genres"])
    # Preprocess items,
    items["item_id"] = pd.to_numeric(items["item_id"])
    ##categories to index
    items["title"], uniques = pd.factorize(items["title"])
    items["genres"], uniques = pd.factorize(items["genres"])
    ##Set category dtype
    items["title"] = items.title.astype("category")
    items["genres"] = items.genres.astype("category")

    return items


@pytest.fixture
def dummy_interaction_dataset(
    dummy_interactions, dummy_user_features, dummy_item_features
):
    return InteractionsDataset(
        dummy_interactions,
        dummy_user_features,
        dummy_item_features,
        interaction_id="rating",
    )