import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    """CatBoost hyperparameters."""

    iterations: int = dataclasses.field(default=1000)
    learning_rate: float = dataclasses.field(default=0.1)
    depth: int = dataclasses.field(default=10)
    l2_leaf_reg: int = dataclasses.field(default=3)
    task_type: str = dataclasses.field(default="CPU")


@dataclasses.dataclass
class DataConfig:
    train_data: str = dataclasses.field(default="data/train.csv")
    val_data: str = dataclasses.field(default="data/val.csv")
    cat_features_indices: list[int] = dataclasses.field(default_factory=lambda: [0])


@dataclasses.dataclass
class ExperimentsConfig:
    dir: str = dataclasses.field(default="experiments")
    save_to: Optional[str] = dataclasses.field(default="latest")


@dataclasses.dataclass
class Config:
    seed: int = dataclasses.field(default=21)
    wandb: Optional[str] = dataclasses.field(default=None)
    data_config: DataConfig = dataclasses.field(default_factory=DataConfig)
    model_config: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    experiments_config: ExperimentsConfig = dataclasses.field(
        default_factory=ExperimentsConfig
    )
