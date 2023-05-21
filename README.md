# 🍷 Wine Quality

A service for determining the quality of wine by its numerical parameters.

## Structure
* `cd` contains Jenkinsfile for setting up cd pipeline;
* `ci` contains Jenkinsfile for setting up ci pipeline;
* `configs` contains the yaml files that define model hyperparameters;
* `data` ‒ the directory with train/val/test datasets;
* `experiments` contains catboost meta info and a model in binary file;
* `notebooks` ‒ the directory with ipynb files with experiments and data analysis;
* `scripts` ‒ executable files (`python -m scripts.<script-name> [ARGS]`);
* `src` ‒ source code of the function used by scripts, also contains unit test of its functions;
* `tests` ‒ functional tests;

## Instalation
Local installation via pip:
```shell
git clone https://github.com/borisshapa/wine-quality
cd wine-quality
pip install -r requirements.txt
```

## Data
For data we are using `.csv` format with `;` as separator. Each line looks like this:
```text
1;6.8;0.11;0.27;8.6;0.044;45.0;104.0;0.99454;3.2;0.37;9.9;6
```

The parameters go in the following order:
0. wine type (categorical feature: 0 ‒ red wine, 1 ‒ white wine)
1. fixed acidity
2. volatile acidity
3. citric acid
4.  residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol
12. **quality** (score between 0 and 10) ‒ target variable.

## Model
The algorithm is based on gradient boosting on decision trees. The service uses the [catboost](https://catboost.ai/) library.
The model is defined by the yaml file. Example:
```yaml
seed: 21
wandb: ~

data:
  train_data: data/train.csv
  val_data: data/val.csv
  cat_features_indices: [0]

model:
  iterations: 1000
  learning_rate: 0.1
  depth: 10
  l2_leaf_reg: 3
  task_type: "CPU"
```

## Training
When training the model on the CPU, it is possible to log metrics and loss in [wandb](https://wandb.ai/site).
So pay attention to the parameter model.task_type ("CPU" or "GPU") in yaml file that defines model hyper parameters.

Also, to send data to wandb, parameter `wandb` must be defined in the yaml config.

If you decide to log data in wandb, log in first:
```shell
wandb login
```

To run training, use `scripts.train`:
```shell
python -m scripts.train --config configs/default.yaml \
    --experiments-dir experiments \
    --save-to latest
```

## Inference
To run model on test dataset use the `scripts.eval`:
```shell
python -m scripts.eval experiments/latest/model.cbm \
    --test-data data/test.csv
```

To train model, evaluate it on test dataset and run unit tests you can use docker compose:
```shell
docker compose up -d
```

To check the results use:
```shell
docker compose logs
```