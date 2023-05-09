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
        default="experiments/09_05_23_13:09:30/model.cbm",
        help="Catboost model in binary file.",
    )
    argparser.add_argument(
        "--test-data",
        type=str,
        default="data/test.csv",
        help="Test dataset in csv format (sep = ';').",
    )
    return argparser


def main(args: argparse.Namespace):
    model = catboost.CatBoostClassifier()

    loguru.logger.info("Loading model from {}", args.model)
    model.load_model(args.model)

    loguru.logger.info("Loading data from {}", args.test_data)
    data = pd.read_csv(args.test_data, sep=";")
    features, labels = utils.split_into_x_y(data)
    predictions = model.predict(features)

    f1_micro = metrics.f1_score(predictions, labels, average="micro")
    accuracy = metrics.accuracy_score(predictions, labels)

    loguru.logger.info(f"\nF1 micro: {f1_micro}\nAccuracy: {accuracy}")


if __name__ == "__main__":
    _args = _configure_parser().parse_args()
    main(_args)
