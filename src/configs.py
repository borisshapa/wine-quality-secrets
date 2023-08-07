import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    """CatBoost hyperparameters."""

    n_estimators: int = dataclasses.field(default=100)
    learning_rate: float = dataclasses.field(default=0.1)
    max_depth: int = dataclasses.field(default=3)


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
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    experiments: ExperimentsConfig = dataclasses.field(
        default_factory=ExperimentsConfig
    )
