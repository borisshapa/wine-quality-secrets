import argparse

import catboost
import loguru
import pandas as pd
from sklearn import metrics

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        default="experiments/latest/model.cbm",
        help="Catboost model in binary file.",
    )
    argparser.add_argument(
        "--test-data",
        type=str,
        default="data/test.csv",
        help="Test dataset in csv format (sep = ';').",
    )
    return argparser


def main(model: str, test_data: str):
    classifier = catboost.CatBoostClassifier()

    loguru.logger.info("Loading model from {}", model)
    classifier.load_model(model)

    loguru.logger.info("Loading data from {}", test_data)
    data = pd.read_csv(test_data, sep=";")
    features, labels = utils.split_into_x_y(data)
    predictions = classifier.predict(features)

    f1_micro = metrics.f1_score(predictions, labels, average="micro")
    accuracy = metrics.accuracy_score(predictions, labels)

    loguru.logger.info(f"\nF1 micro: {f1_micro}\nAccuracy: {accuracy}")


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(**vars(args))
