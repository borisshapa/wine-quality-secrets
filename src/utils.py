import datetime
import os
import random

import numpy as np
import pandas as pd
from sklearn import model_selection

WINE_TYPE_COLUMN_NAME = "wine type"
CSV_SEPARATOR = ";"


def split_into_train_val_test(
    data: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, pd.DataFrame]:
    train, test = model_selection.train_test_split(
        data, test_size=test_ratio, random_state=seed
    )

    train, val = model_selection.train_test_split(
        train, test_size=val_ratio / (1 - test_ratio), random_state=seed
    )

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def set_deterministic_mode(seed: int):
    """Fixes the seed for all random processes
    (for pure python, numpy).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_into_x_y(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    return x, y


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%d_%m_%y_%H:%M:%S")
