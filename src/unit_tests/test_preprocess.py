import unittest

import numpy as np

from src.utils import common


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        features = 7
        classes = 5
        data_size = 10
        self.x = np.random.rand(data_size, features)
        self.y = np.random.randint(classes, size=data_size)

    def test_split_into_train_val_test__sizes(self):
        val_ratio = 0.2
        test_ratio = 0.5

        partition = common.split_into_train_val_test(
            self.x, self.y, val_ratio, test_ratio
        )

        self.assertEqual(len(partition["train"][1]), 3)
        self.assertEqual(len(partition["val"][1]), 2)
        self.assertEqual(len(partition["test"][1]), 5)

        self.assertEqual(len(partition["train"][0]), len(partition["train"][1]))
        self.assertEqual(len(partition["val"][0]), len(partition["val"][1]))
        self.assertEqual(len(partition["test"][0]), len(partition["test"][1]))
