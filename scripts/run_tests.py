import argparse
import os

import catboost
import loguru
import ujson


def _configure_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        default="experiments/latest/model.cbm",
        help="the binary catboost model file",
    )
    argparser.add_argument(
        "--tests-dir",
        type=str,
        default="tests",
        help="the directory with jsons contain tests",
    )
    return argparser


def main(model: str, tests_dir: str):
    classifier = catboost.CatBoostClassifier()

    loguru.logger.info("Loading model from {}", model)
    classifier.load_model(model)

    total = 0
    success = 0

    for f in os.listdir(tests_dir):
        full_path = os.path.join(tests_dir, f)
        if os.path.isfile(full_path):
            loguru.logger.info("Running tests from {}", f)
            with open(full_path, "r", encoding="utf-8") as json_file:
                tests = ujson.load(json_file)

                features = [list(id2value.values()) for id2value in tests["X"]]
                targets = [list(id2value.values()) for id2value in tests["y"]]

                predictions = classifier.predict(features)

                for i in range(len(features)):
                    loguru.logger.info("TEST {} | features: {}", i, features[i])
                    total += 1
                    if predictions[i] != targets[i]:
                        loguru.logger.warning(
                            "FAILED | Expected: {}, found: {}",
                            targets[i],
                            predictions[i],
                        )
                    else:
                        loguru.logger.info("SUCCESS")
                        success += 1

    loguru.logger.info("{}/{} test accepted", success, total)


if __name__ == "__main__":
    args = _configure_argparser().parse_args()
    main(**vars(args))
