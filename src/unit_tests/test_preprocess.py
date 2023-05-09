import unittest

import numpy as np
import pandas as pd
from pandas import testing

from src import utils


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        data_size = 10
        self.data = pd.DataFrame(
            data={
                "col1": np.random.rand(data_size),
                "col2": np.random.rand(data_size),
                "col3": np.random.rand(data_size),
            }
        )

    def test_split_into_train_val_test__sizes(self):
        val_ratio = 0.2
        test_ratio = 0.5

        partition = utils.split_into_train_val_test(
            self.data, val_ratio, test_ratio
        )

        self.assertEqual(len(partition["train"].index), 3)
        self.assertEqual(len(partition["val"].index), 2)
        self.assertEqual(len(partition["test"].index), 5)

    def test_split_into_x_y(self):
        x, y = utils.split_into_x_y(self.data)
        testing.assert_frame_equal(x, self.data[["col1", "col2"]])
        testing.assert_series_equal(y, self.data["col3"])
