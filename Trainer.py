import argparse
import logging
import time
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from Variable import Variable
from MLP import MLP
from AdamW import AdamW
from Dataset import PolynomialDataset


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
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train the model.

        Args:
            dataset: Dataset to train on
            epochs: Number of epochs to train
            validate_every: Validate every N epochs
            verbose: Whether to print progress

        Returns:
            List of training metrics per epoch
        """
        metrics = []

        start_time = time.time()
        for epoch in range(epochs + 1):
            epoch_metrics = {}

            # Train one epoch (skip for epoch 0)
            if epoch > 0:
                train_loss = self.train_epoch(dataset, verbose=False)
                epoch_metrics["train_loss"] = train_loss

            # Validate
            if epoch % validate_every == 0 or epoch == epochs:
                eval_loss = self.evaluate(dataset)
                epoch_metrics["eval_loss"] = eval_loss

                if verbose:
                    elapsed = time.time() - start_time
                    metrics_str = ", ".join(
                        [f"{k}: {v:.6f}" for k, v in epoch_metrics.items()])
                    print(
                        f"Epoch {epoch}/{epochs}, {metrics_str}, time: {elapsed:.2f}s")

            metrics.append(epoch_metrics)

        return metrics


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description="Train an MLP model")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[16, 16],
                        help="Sizes of hidden layers")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay factor")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Noise level for dataset")
    parser.add_argument("--polynomial-coeffs", type=float, nargs="+", default=[10, -3, -2, 1],
                        help="Coefficients of the polynomial, from highest to lowest power")

    args = parser.parse_args()

    # Create dataset
    dataset = PolynomialDataset(args.polynomial_coeffs, noise_std=args.noise)
    # Format the polynomial as a string for display
    poly_terms = []
    for i, coeff in enumerate(args.polynomial_coeffs):
        power = len(args.polynomial_coeffs) - i - 1
        if coeff == 0:
            continue
        if power == 0:
            term = f"{coeff}"
        elif power == 1:
            term = f"{coeff}x"
        else:
            term = f"{coeff}x^{power}"
        poly_terms.append(term)
    poly_str = " + ".join(poly_terms).replace(" + -", " - ")

    print(f"Created polynomial dataset: {poly_str}")

    # Create model (1 input feature, specified hidden layers, 1 output)
    model = MLP(1, args.hidden_sizes, 1)
    print(
        f"Created MLP with architecture: 1 -> {' -> '.join(map(str, args.hidden_sizes))} -> 1")

    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )

    # Train model
    print(
        f"Training for {args.epochs} epochs with lr={args.lr}, batch_size={args.batch_size}")
    metrics = trainer.train(dataset, epochs=args.epochs)

    # Final evaluation
    final_loss = trainer.evaluate(dataset)
    print(f"Final evaluation loss: {final_loss:.6f}")

    # Visualize results
    visualize_predictions(model, dataset)


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

    step = (x_max - x_min) / (num_points - 1)
    x_range = [x_min + i * step for i in range(num_points)]

    predictions = []
    for x in x_range:
        output = model([Variable(x)])
        predictions.append(output[0].data)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', label='Test data')
    plt.plot(x_range, predictions, color='red', label='Model predictions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Model Predictions vs Test Data')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
