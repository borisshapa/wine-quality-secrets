import dataclasses
import os.path

import catboost
import loguru
import pandas as pd
import wandb

from src import utils, callbacks, configs


def train(config: configs.Config):
    utils.set_deterministic_mode(config.seed)
    data_config = config.data_config
    model_config = config.model_config
    experiments_config = config.experiments_config

    use_wandb = model_config.task_type == "CPU" and config.wandb is not None

    if use_wandb:
        wandb.init(
            project=config.wandb,
            config=dataclasses.asdict(config),
        )

    loguru.logger.info("Loading train data from {}", data_config.train_data)
    train_data = pd.read_csv(data_config.train_data, sep=utils.CSV_SEPARATOR)

    loguru.logger.info("Loading val data from {}", data_config.val_data)
    val_data = pd.read_csv(data_config.val_data, sep=utils.CSV_SEPARATOR)

    x_train, y_train = utils.split_into_x_y(train_data)
    x_val, y_val = utils.split_into_x_y(val_data)

    train_dir = os.path.join(
        experiments_config.dir, experiments_config.save_to or utils.get_current_time()
    )
    loguru.logger.info("Training... | train dir: {}", train_dir)

    model = catboost.CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Micro",
        random_seed=config.seed,
        train_dir=train_dir,
        **dataclasses.asdict(model_config)
    )

    verbose = True
    _callbacks = None

    if use_wandb:
        verbose = False
        _callbacks = [callbacks.WAndBCallback(model_config.iterations)]

    model.fit(
        x_train,
        y_train,
        eval_set=(x_val, y_val),
        cat_features=data_config.cat_features_indices,
        verbose=verbose,
        callbacks=_callbacks,
        plot=False,
    )

    save_to = os.path.join(train_dir, "model.cbm")
    loguru.logger.info("Saving model to {}", save_to)
    model.save_model(save_to)
