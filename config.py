"""
Configuration management for the neural network framework.

This module provides a simple configuration system for managing hyperparameters
and training settings in a structured way.
"""

from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    input_size: int
    hidden_sizes: list
    output_size: int
    activation: str = "relu"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.input_size <= 0:
            raise ValueError(
                f"input_size must be positive, got {self.input_size}")
        if self.output_size <= 0:
            raise ValueError(
                f"output_size must be positive, got {self.output_size}")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError(
                f"All hidden sizes must be positive, got {self.hidden_sizes}"
            )
        if self.activation not in ["relu", "tanh", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""

    learning_rate: float = 0.001
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.eps < 0:
            raise ValueError(f"eps must be non-negative, got {self.eps}")
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError(f"betas must be in [0, 1), got {self.betas}")
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training settings."""

    epochs: int = 100
    batch_size: int = 64
    validate_every: int = 10
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-6
    loss_fn: str = "mse"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}")
        if self.validate_every <= 0:
            raise ValueError(
                f"validate_every must be positive, got {self.validate_every}"
            )
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )
        if self.loss_fn not in ["mse", "mae"]:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")


@dataclass
class DatasetConfig:
    """Configuration for dataset settings."""

    coefficients: list
    domain: tuple = (-1.0, 1.0)
    num_samples: int = 10000
    noise_std: float = 0.1
    test_ratio: float = 0.2
    normalize_inputs: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.coefficients:
            raise ValueError("coefficients cannot be empty")
        if len(self.domain) != 2 or self.domain[1] <= self.domain[0]:
            raise ValueError(
                f"domain must be (min, max) with min < max, got {self.domain}"
            )
        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {self.num_samples}")
        if self.noise_std < 0:
            raise ValueError(
                f"noise_std must be non-negative, got {self.noise_std}")
        if not 0 <= self.test_ratio <= 1:
            raise ValueError(
                f"test_ratio must be between 0 and 1, got {self.test_ratio}"
            )


@dataclass
class Config:
    """Main configuration class combining all settings."""

    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    dataset: DatasetConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            Config instance
        """
        return cls(
            model=ModelConfig(**config_dict["model"]),
            optimizer=OptimizerConfig(**config_dict["optimizer"]),
            training=TrainingConfig(**config_dict["training"]),
            dataset=DatasetConfig(**config_dict["dataset"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
            "dataset": asdict(self.dataset),
        }

    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save configuration
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            Config instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> Config:
    """
    Get default configuration for polynomial regression.

    Returns:
        Default Config instance
    """
    return Config(
        model=ModelConfig(
            input_size=1, hidden_sizes=[16, 16], output_size=1, activation="relu"
        ),
        optimizer=OptimizerConfig(learning_rate=0.01, weight_decay=0.01),
        training=TrainingConfig(
            epochs=100, batch_size=64, validate_every=10, early_stopping_patience=10
        ),
        dataset=DatasetConfig(
            # x^2 - 1
            coefficients=[1, 0, -1], num_samples=10000, noise_std=0.1
        ),
    )
