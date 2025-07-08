import argparse
import logging
import time
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from variable import Variable
from mlp import MLP
from adam_w import AdamW
from dataset import PolynomialDataset


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mean_squared_error(predictions: List[Variable], targets: List[Variable]) -> Variable:
    """
    Compute mean squared error between predictions and targets.
    
    Args:
        predictions: List of predicted Variables
        targets: List of target Variables
        
    Returns:
        MSE loss as a Variable
        
    Raises:
        ValueError: If predictions and targets have different lengths
        
    Examples:
        >>> pred = [Variable(1.0), Variable(2.0)]
        >>> target = [Variable(1.1), Variable(1.9)]
        >>> loss = mean_squared_error(pred, target)
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) doesn't match targets ({len(targets)})")

    if len(predictions) == 0:
        raise ValueError("Cannot compute MSE for empty lists")

    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def mean_absolute_error(predictions: List[Variable], targets: List[Variable]) -> Variable:
    """
    Compute mean absolute error between predictions and targets.
    
    Args:
        predictions: List of predicted Variables
        targets: List of target Variables
        
    Returns:
        MAE loss as a Variable
        
    Raises:
        ValueError: If predictions and targets have different lengths
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) doesn't match targets ({len(targets)})")

    if len(predictions) == 0:
        raise ValueError("Cannot compute MAE for empty lists")

    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)


def r2_score(predictions: List[Variable], targets: List[Variable]) -> float:
    """
    Compute R-squared score for regression evaluation.
    
    Args:
        predictions: List of predicted Variables  
        targets: List of target Variables
        
    Returns:
        R² score as a float
        
    Raises:
        ValueError: If predictions and targets have different lengths
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) doesn't match targets ({len(targets)})")

    if len(predictions) == 0:
        raise ValueError("Cannot compute R² for empty lists")

    # Convert to data values for computation
    y_true = [t.data for t in targets]
    y_pred = [p.data for p in predictions]
    
    # Calculate means
    y_mean = sum(y_true) / len(y_true)
    
    # Calculate sums of squares
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    
    # Handle edge case where all targets are the same
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - (ss_res / ss_tot)


class Trainer:
    """
    Class to handle model training and evaluation.
    
    Provides functionality for training neural networks with configurable
    optimizers, loss functions, and evaluation metrics.
    
    Examples:
        >>> model = MLP(input_size=1, hidden_sizes=[16], output_size=1)
        >>> trainer = Trainer(model, learning_rate=0.01)
        >>> dataset = PolynomialDataset([1, 0, -1])  # x^2 - 1
        >>> metrics = trainer.train(dataset, epochs=100)
    """

    def __init__(
        self,
        model: MLP,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        weight_decay: float = 0.01,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            learning_rate: Learning rate for optimization
            batch_size: Number of samples per batch
            weight_decay: Weight decay factor
            
        Raises:
            ValueError: If hyperparameters are invalid
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        logger.info(f"Initialized trainer with lr={learning_rate}, "
                   f"batch_size={batch_size}, weight_decay={weight_decay}")
        logger.info(f"Model: {model}")

    def train_epoch(self, dataset, verbose: bool = False, 
                   loss_fn: str = 'mse') -> float:
        """
        Train for one epoch.

        Args:
            dataset: Dataset to train on
            verbose: Whether to log batch losses
            loss_fn: Loss function to use ('mse' or 'mae')

        Returns:
            Average loss for the epoch
            
        Raises:
            ValueError: If loss_fn is not supported
        """
        if loss_fn not in ['mse', 'mae']:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
        
        loss_func = mean_squared_error if loss_fn == 'mse' else mean_absolute_error
        
        num_batches = max(1, len(dataset) // self.batch_size)
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            try:
                # Get batch
                batch_x, batch_y = dataset.get_batch(self.batch_size)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                batch_loss = Variable(0.0)
                for x, y in zip(batch_x, batch_y):
                    output = self.model(x)
                    batch_loss = batch_loss + loss_func(output, y)

                # Average batch loss
                batch_loss = batch_loss / len(batch_x)

                # Backward pass
                batch_loss.backward()

                # Gradient clipping for stability
                for param in self.model.parameters():
                    param.clip_grad(max_norm=1.0)

                # Update parameters
                self.optimizer.step()

                # Track loss
                epoch_loss += batch_loss.data

                if verbose and batch_idx % 10 == 0:
                    logger.info(f"  Batch {batch_idx}/{num_batches}, loss: {batch_loss.data:.6f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                raise

        # Return average loss
        return epoch_loss / num_batches

    def evaluate(self, dataset, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model on dataset with multiple metrics.

        Args:
            dataset: Dataset to evaluate on
            metrics: List of metrics to compute ['mse', 'mae', 'r2']

        Returns:
            Dictionary of metric names to values
        """
        if metrics is None:
            metrics = ['mse']
            
        # Validate metrics
        valid_metrics = ['mse', 'mae', 'r2']
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid: {valid_metrics}")
        
        if hasattr(dataset, "get_test_data"):
            # Use test data if available
            test_x, test_y = dataset.get_test_data()
        else:
            # Use all data
            test_x, test_y = [], []
            for i in range(len(dataset)):
                x, y = dataset[i]
                test_x.append(x)
                test_y.append(y)

        if len(test_x) == 0:
            raise ValueError("No data available for evaluation")

        # Get predictions
        predictions = []
        targets = []
        for x, y in zip(test_x, test_y):
            output = self.model(x)
            predictions.extend(output)
            targets.extend(y)
        
        # Compute metrics
        results = {}
        if 'mse' in metrics:
            mse_loss = mean_squared_error(predictions, targets)
            results['mse'] = mse_loss.data
        
        if 'mae' in metrics:
            mae_loss = mean_absolute_error(predictions, targets)
            results['mae'] = mae_loss.data
            
        if 'r2' in metrics:
            results['r2'] = r2_score(predictions, targets)

        return results

    def train(
        self,
        dataset,
        epochs: int = 100,
        validate_every: int = 10,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-6,
        loss_fn: str = 'mse'
    ) -> List[Dict[str, float]]:
        """
        Train the model with optional early stopping.

        Args:
            dataset: Dataset to train on
            epochs: Number of epochs to train
            validate_every: Validate every N epochs
            verbose: Whether to log progress
            early_stopping_patience: Stop if no improvement for N validations (None to disable)
            early_stopping_min_delta: Minimum change to qualify as improvement
            loss_fn: Loss function to use ('mse' or 'mae')

        Returns:
            List of training metrics per epoch
            
        Raises:
            ValueError: If parameters are invalid
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if validate_every <= 0:
            raise ValueError(f"validate_every must be positive, got {validate_every}")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {early_stopping_patience}")
        
        metrics = []
        best_eval_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        logger.info(f"Starting training for {epochs} epochs")
        if early_stopping_patience:
            logger.info(f"Early stopping enabled with patience={early_stopping_patience}")

        start_time = time.time()
        for epoch in range(epochs + 1):
            epoch_metrics = {}

            # Train one epoch (skip for epoch 0)
            if epoch > 0:
                try:
                    train_loss = self.train_epoch(dataset, verbose=verbose, loss_fn=loss_fn)
                    epoch_metrics["train_loss"] = train_loss
                except Exception as e:
                    logger.error(f"Training failed at epoch {epoch}: {e}")
                    raise

            # Validate
            if epoch % validate_every == 0 or epoch == epochs:
                try:
                    eval_metrics = self.evaluate(dataset, metrics=['mse', 'mae', 'r2'])
                    epoch_metrics.update(eval_metrics)
                    
                    # Early stopping logic
                    current_eval_loss = eval_metrics['mse']  # Use MSE for early stopping
                    if early_stopping_patience:
                        if current_eval_loss < best_eval_loss - early_stopping_min_delta:
                            best_eval_loss = current_eval_loss
                            best_epoch = epoch
                            patience_counter = 0
                            logger.info(f"New best validation loss: {best_eval_loss:.6f}")
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered at epoch {epoch}. "
                                      f"Best epoch: {best_epoch} with loss: {best_eval_loss:.6f}")
                            break

                    if verbose:
                        elapsed = time.time() - start_time
                        metrics_str = ", ".join(
                            [f"{k}: {v:.6f}" for k, v in epoch_metrics.items()])
                        logger.info(f"Epoch {epoch}/{epochs}, {metrics_str}, time: {elapsed:.2f}s")
                        
                except Exception as e:
                    logger.error(f"Evaluation failed at epoch {epoch}: {e}")
                    raise

            metrics.append(epoch_metrics)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        return metrics


def main():
    """Main function to run training with enhanced configuration support."""
    parser = argparse.ArgumentParser(description="Train an MLP model for polynomial regression")
    
    # Model configuration
    parser.add_argument("--input-size", type=int, default=1,
                        help="Number of input features")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[16, 16],
                        help="Sizes of hidden layers")
    parser.add_argument("--output-size", type=int, default=1,
                        help="Number of output features")
    parser.add_argument("--activation", type=str, default="relu", 
                        choices=['relu', 'tanh', 'sigmoid'],
                        help="Activation function")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--validate-every", type=int, default=10,
                        help="Validate every N epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Early stopping patience (None to disable)")
    parser.add_argument("--loss-fn", type=str, default="mse", choices=['mse', 'mae'],
                        help="Loss function to use")
    
    # Optimizer configuration
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay factor")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta2 for AdamW optimizer")
    
    # Dataset configuration
    parser.add_argument("--polynomial-coeffs", type=float, nargs="+", default=[1, 0, -1],
                        help="Coefficients of the polynomial, from highest to lowest power")
    parser.add_argument("--domain", type=float, nargs=2, default=[-1.0, 1.0],
                        help="Domain for input values (min max)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise level for dataset")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--normalize-inputs", action="store_true",
                        help="Normalize input values to [-1, 1]")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible results")
    
    # Output configuration
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save trained model")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load pretrained model")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--save-config", type=str, default=None,
                        help="Path to save configuration")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Load configuration from file if provided
        if args.config_file:
            from config import Config
            config = Config.load(args.config_file)
            logger.info(f"Loaded configuration from {args.config_file}")
        else:
            # Create configuration from command line arguments
            from config import Config, ModelConfig, OptimizerConfig, TrainingConfig, DatasetConfig
            config = Config(
                model=ModelConfig(
                    input_size=args.input_size,
                    hidden_sizes=args.hidden_sizes,
                    output_size=args.output_size,
                    activation=args.activation
                ),
                optimizer=OptimizerConfig(
                    learning_rate=args.lr,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay
                ),
                training=TrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    validate_every=args.validate_every,
                    early_stopping_patience=args.early_stopping_patience,
                    loss_fn=args.loss_fn
                ),
                dataset=DatasetConfig(
                    coefficients=args.polynomial_coeffs,
                    domain=tuple(args.domain),
                    num_samples=args.num_samples,
                    noise_std=args.noise,
                    test_ratio=args.test_ratio,
                    normalize_inputs=args.normalize_inputs,
                    seed=args.seed
                )
            )

        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            logger.info(f"Saved configuration to {args.save_config}")

        # Create dataset
        dataset = PolynomialDataset(
            coefficients=config.dataset.coefficients,
            domain=config.dataset.domain,
            num_samples=config.dataset.num_samples,
            noise_std=config.dataset.noise_std,
            test_ratio=config.dataset.test_ratio,
            normalize_inputs=config.dataset.normalize_inputs,
            seed=config.dataset.seed
        )
        
        logger.info(f"Created dataset: {dataset}")

        # Load or create model
        if args.load_model:
            from model_utils import load_model, create_optimizer_from_model
            model, optimizer_state, saved_config, metadata = load_model(args.load_model)
            optimizer = create_optimizer_from_model(model, optimizer_state)
            logger.info(f"Loaded model from {args.load_model}")
            if metadata:
                logger.info(f"Model metadata: {metadata}")
        else:
            # Create model
            model = MLP(
                input_size=config.model.input_size,
                hidden_sizes=config.model.hidden_sizes,
                output_size=config.model.output_size,
                activation=config.model.activation
            )
            logger.info(f"Created model: {model}")

            # Create trainer
            trainer = Trainer(
                model,
                learning_rate=config.optimizer.learning_rate,
                batch_size=config.training.batch_size,
                weight_decay=config.optimizer.weight_decay
            )

            # Train model
            logger.info("Starting training...")
            metrics = trainer.train(
                dataset,
                epochs=config.training.epochs,
                validate_every=config.training.validate_every,
                verbose=args.verbose,
                early_stopping_patience=config.training.early_stopping_patience,
                loss_fn=config.training.loss_fn
            )

            # Display final results
            if metrics:
                final_metrics = metrics[-1]
                logger.info(f"Training completed. Final metrics: {final_metrics}")

            # Save model if requested
            if args.save_model:
                from model_utils import save_model
                metadata = {
                    'final_metrics': final_metrics if metrics else {},
                    'training_history': metrics,
                    'dataset_info': str(dataset)
                }
                save_model(model, args.save_model, trainer.optimizer, config, metadata)
                logger.info(f"Model saved to {args.save_model}")

        # Visualization (if not loading model)
        if not args.load_model:
            try:
                visualize_predictions(model, dataset)
                logger.info("Generated prediction visualization")
            except Exception as e:
                logger.warning(f"Could not generate visualization: {e}")

    except Exception as e:
        logger.error(f"Training failed: {e}")

def visualize_predictions(model, dataset):
    """
    Plot test data and model predictions.

    Args:
        model: Trained model
        dataset: Dataset with test data
    """
    # Get test data
    if hasattr(dataset, "get_test_data"):
        test_x, test_y = dataset.get_test_data()
    else:
        # Use all data
        test_x, test_y = [], []
        for i in range(len(dataset)):
            x, y = dataset[i]
            test_x.append(x)
            test_y.append(y)

    if not test_x:
        logger.warning("No test data available for visualization")
        return

    # Sort data points by x value for smooth curve
    sorted_data = [(x[0].data, y[0].data) for x, y in zip(test_x, test_y)]
    sorted_data.sort(key=lambda point: point[0])

    # Extract sorted data
    x_values = [point[0] for point in sorted_data]
    y_values = [point[1] for point in sorted_data]

    # Generate model predictions
    x_min = min(x_values)
    x_max = max(x_values)
    num_points = 100

    step = (x_max - x_min) / (num_points - 1) if num_points > 1 else 0
    x_range = [x_min + i * step for i in range(num_points)]

    predictions = []
    for x in x_range:
        output = model([Variable(x)])
        predictions.append(output[0].data)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', alpha=0.6, label='Test data')
    plt.plot(x_range, predictions, color='red', linewidth=2, label='Model predictions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Model Predictions vs Test Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
