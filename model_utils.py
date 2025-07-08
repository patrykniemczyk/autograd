"""
Utilities for saving and loading models.

This module provides simple functionality to save and load neural network models
along with their training state and configuration.
"""

import pickle
from typing import Dict, Any, Optional
from pathlib import Path

from variable import Variable
from mlp import MLP
from adam_w import AdamW
from config import Config


def save_model(
    model: MLP,
    filepath: str,
    optimizer: Optional[AdamW] = None,
    config: Optional[Config] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model state to file.

    Args:
        model: The MLP model to save
        filepath: Path to save the model
        optimizer: Optional optimizer state to save
        config: Optional configuration to save
        metadata: Optional metadata (training metrics, etc.)
    """
    # Extract model parameters
    model_state = {
        "input_size": model.input_size,
        "hidden_sizes": model.hidden_sizes,
        "output_size": model.output_size,
        "activation_name": model.activation_name,
        "parameters": [param.data for param in model.parameters()],
    }

    # Create save dictionary
    save_dict = {
        "model_state": model_state,
        "optimizer_state": optimizer.get_state_dict() if optimizer else None,
        "config": config.to_dict() if config else None,
        "metadata": metadata or {},
    }

    # Save to file
    with open(filepath, "wb") as f:
        pickle.dump(save_dict, f)


def load_model(filepath: str) -> tuple:
    """
    Load model from file.

    Args:
        filepath: Path to the saved model

    Returns:
        Tuple of (model, optimizer_state, config, metadata)
    """
    with open(filepath, "rb") as f:
        save_dict = pickle.load(f)

    # Reconstruct model
    model_state = save_dict["model_state"]
    model = MLP(
        input_size=model_state["input_size"],
        hidden_sizes=model_state["hidden_sizes"],
        output_size=model_state["output_size"],
        activation=model_state["activation_name"],
    )

    # Load parameter values
    params = model.parameters()
    param_values = model_state["parameters"]

    if len(params) != len(param_values):
        raise ValueError(
            f"Parameter count mismatch: expected {len(params)}, got {len(param_values)}"
        )

    for param, value in zip(params, param_values):
        param.data = value

    # Load other components
    optimizer_state = save_dict.get("optimizer_state")
    config_dict = save_dict.get("config")
    config = Config.from_dict(config_dict) if config_dict else None
    metadata = save_dict.get("metadata", {})

    return model, optimizer_state, config, metadata


def create_optimizer_from_model(
    model: MLP, optimizer_state: Optional[Dict] = None
) -> AdamW:
    """
    Create optimizer for a model, optionally loading saved state.

    Args:
        model: The MLP model
        optimizer_state: Optional saved optimizer state

    Returns:
        AdamW optimizer
    """
    if optimizer_state:
        # Create optimizer with saved hyperparameters
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_state["lr"],
            betas=optimizer_state["betas"],
            eps=optimizer_state["eps"],
            weight_decay=optimizer_state["weight_decay"],
        )
        # Load state
        optimizer.load_state_dict(optimizer_state)
    else:
        # Create with default parameters
        optimizer = AdamW(model.parameters())

    return optimizer


def export_model_info(model: MLP, filepath: str) -> None:
    """
    Export human-readable model information to text file.

    Args:
        model: The MLP model
        filepath: Path to save the info file
    """
    with open(filepath, "w") as f:
        f.write(f"Model Architecture\n")
        f.write(f"==================\n")
        f.write(f"Type: Multi-Layer Perceptron\n")
        f.write(f"Input size: {model.input_size}\n")
        f.write(f"Hidden layers: {model.hidden_sizes}\n")
        f.write(f"Output size: {model.output_size}\n")
        f.write(f"Activation: {model.activation_name}\n")
        f.write(f"Total parameters: {model.num_parameters()}\n\n")

        f.write(f"Layer Details\n")
        f.write(f"=============\n")
        layer_idx = 0
        param_idx = 0

        # Analyze layer structure
        layer_sizes = [model.input_size] + \
            model.hidden_sizes + [model.output_size]

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            layer_params = input_size * output_size + output_size  # weights + biases

            f.write(f"Linear Layer {layer_idx + 1}:\n")
            f.write(f"  Input size: {input_size}\n")
            f.write(f"  Output size: {output_size}\n")
            f.write(f"  Parameters: {layer_params}\n")
            f.write(
                f"  Parameter range: {param_idx} - {param_idx + layer_params - 1}\n"
            )

            param_idx += layer_params
            layer_idx += 1

            # Add activation info if not the last layer
            if i < len(layer_sizes) - 2:
                f.write(f"\n{model.activation_name.upper()} Activation\n")
                f.write(f"  Parameters: 0\n")

            f.write(f"\n")


# Example usage functions
def save_training_checkpoint(
    model: MLP,
    optimizer: AdamW,
    config: Config,
    metrics: list,
    epoch: int,
    checkpoint_dir: str = "checkpoints",
) -> str:
    """
    Save a training checkpoint.

    Args:
        model: The MLP model
        optimizer: The optimizer
        config: Training configuration
        metrics: Training metrics history
        epoch: Current epoch
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Path to saved checkpoint
    """
    Path(checkpoint_dir).mkdir(exist_ok=True)

    metadata = {
        "epoch": epoch,
        "metrics": metrics,
        "timestamp": __import__("time").time(),
    }

    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pkl"
    save_model(model, checkpoint_path, optimizer, config, metadata)

    return checkpoint_path


def load_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> tuple:
    """
    Load the latest checkpoint from directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Tuple of (model, optimizer_state, config, metadata) or (None,) * 4 if no checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, None, None, None

    # Find latest checkpoint
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pkl"))
    if not checkpoint_files:
        return None, None, None, None

    # Sort by epoch number
    latest_file = max(checkpoint_files,
                      key=lambda f: int(f.stem.split("_")[-1]))

    return load_model(str(latest_file))
