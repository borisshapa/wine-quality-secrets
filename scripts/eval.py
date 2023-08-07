import argparse
import pickle

import loguru
from sklearn import metrics

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        default="experiments/latest/model.sklrn",
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
    loguru.logger.info("Loading model from {}", model)

    with open(model, "rb") as model_file:
        classifier = pickle.load(model_file)

    loguru.logger.info("Loading data from {}", test_data)
    x, y, _ = utils.load_data_from_csv(test_data, sep=utils.CSV_SEPARATOR)
    preds = classifier.predict(x)

    f1_micro = metrics.f1_score(y, preds, average="micro")
    accuracy = metrics.accuracy_score(y, preds)

    loguru.logger.info(f"\nF1 micro: {f1_micro}\nAccuracy: {accuracy}")


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(**vars(args))
