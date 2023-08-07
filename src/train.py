import dataclasses
import os.path
import pickle

import loguru
from sklearn import ensemble, metrics

from src import utils, configs


def train(config: configs.Config):
    utils.set_deterministic_mode(config.seed)

    data_config = config.data
    model_config = config.model
    experiments_config = config.experiments

    loguru.logger.info("Loading train data from {}", data_config.train_data)
    x_train, y_train, _ = utils.load_data_from_csv(
        data_config.train_data, sep=utils.CSV_SEPARATOR
    )

    loguru.logger.info("Loading val data from {}", data_config.val_data)
    x_val, y_val, _ = utils.load_data_from_csv(
        data_config.val_data, sep=utils.CSV_SEPARATOR
    )

    train_dir = os.path.join(
        experiments_config.dir, experiments_config.save_to or utils.get_current_time()
    )
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    loguru.logger.info("Training...")

    model = ensemble.GradientBoostingClassifier(
        random_state=config.seed, **dataclasses.asdict(model_config)
    )
    model.fit(x_train, y_train)

    pred_val = model.predict(x_val)
    loguru.logger.info("Val accuracy: {}", metrics.accuracy_score(y_val, pred_val))
    loguru.logger.info(
        "Val f1 micro: {}", metrics.f1_score(y_val, pred_val, average="micro")
    )

    save_to = os.path.join(train_dir, "model.sklrn")
    loguru.logger.info("Saving model to {}", save_to)
    with open(save_to, "wb") as model_file:
        pickle.dump(model, model_file)
