import argparse
import datetime
import os.path

import catboost
import omegaconf
import pandas as pd
import wandb

from src import utils, callbacks


def _configure_arg_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="configs/default.yaml")
    argparser.add_argument("--experiments-dir", type=str, default="experiments")
    return argparser


def main(args: argparse.Namespace):
    yaml_config = omegaconf.OmegaConf.load(args.config)

    utils.set_deterministic_mode(yaml_config["seed"])
    data_config = yaml_config["data"]
    model_config = yaml_config["model"]

    if model_config["task_type"] == "CPU":
        wandb.init(
            project=yaml_config["wandb"],
            config=omegaconf.OmegaConf.to_container(model_config, resolve=True),
        )

    train_data = pd.read_csv(data_config["train_data"], sep=utils.CSV_SEPARATOR)
    val_data = pd.read_csv(data_config["val_data"], sep=utils.CSV_SEPARATOR)

    x_train, y_train = utils.split_into_x_y(train_data)
    x_val, y_val = utils.split_into_x_y(val_data)

    train_dir = os.path.join(
        args.experiments_dir, utils.get_current_time()
    )

    model = catboost.CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Micro",
        random_seed=yaml_config["seed"],
        train_dir=train_dir,
        **model_config
    )

    verbose = True
    _callbacks = None

    if model_config["task_type"] == "CPU":
        verbose = False
        _callbacks = [callbacks.WAndBCallback(model_config["iterations"])]

    model.fit(
        x_train,
        y_train,
        eval_set=(x_val, y_val),
        cat_features=data_config["cat_features_indices"],
        verbose=verbose,
        callbacks=_callbacks,
        plot=False,
    )

    model.save_model(os.path.join(train_dir, "model.cbm"))


if __name__ == "__main__":
    _args = _configure_arg_parser().parse_args()
    main(_args)
